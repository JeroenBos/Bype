from keyboard._0_types import Input, SwipeDataFrame
from keyboard._1_import import keyboards, raw_data
from utilities import concat, get_resource, is_list_of, read_all, skip, split_at, split_by, windowed_2
from typing import List, Union
import drawSvg as draw


def animate_from_frames(frames, dt: float):
    """:param dt: in seconds """
    for i, frame in enumerate(frames):
        animation = dict(
            from_or_values='none;inline;none;none',
            keyTimes=f"0;{i / len(frames):.2f};{(i + 1) / len(frames):.2f};1",  # keyTimes relative to `dur`, and must be in [0, 1]
            attributeName='display',
            begin='0s',
            dur=f"{dt * len(frames):.2f}s",
            repeatCount="indefinite",
        )

        frame.appendAnim(draw.Animate(**animation))
    return frames


def text_file_to_swipes(path: str) -> List[str]:
    raw_data_text = read_all(path)
    return text_to_swipes(raw_data_text)

def text_to_swipes(raw_data_text: str) -> List[str]:

    def split_punctuation(s: str):
        assert isinstance(s, str)
        s = s.replace(' ', '').replace('\r', '')
        split_indices = concat((i, i + 1) for i, c in enumerate(s) if not c.isalpha())

        return split_at(s, *split_indices)

    def is_valid(s: str):
        return len(s) != 0 and not s.isspace()

    words = split_by(raw_data_text, '\n', ' ')  # word here means more generic than string of characters: it's more of a string of values (values represented on keys)
    words = concat(split_punctuation(word) for word in words)
    words = [word for word in words if is_valid(word)]
    return words


def to_frame(swipe: SwipeDataFrame):


    if len(swipe.X) == 0:
        raise ValueError('empty swipe')

    path = draw.Path(stroke_width=20, stroke='green', fill='transparent')
    path.M(swipe.X.iloc[0], swipe.Y.iloc[0])
    for x, y in skip(zip(swipe.X, swipe.Y), 1):
        path.L(x, y)
    return path


def to_svg(elements):
    width, height = 1000, 1000

    root = draw.Group(transform=f'scale(1,-1)')
    root.extend(elements)
    svg = draw.Drawing(width, height, origin=(0, -height))
    svg.append(root)
    return svg


def to_frames(swipes: List[SwipeDataFrame]):

    frames = [to_frame(swipe) for swipe in swipes]
    animate_from_frames(frames, .1)

    return to_svg(frames)


svg = to_frames(raw_data)
svg.saveSvg(get_resource('2020-03-20_0 all words.svg'))


# for i, swipe in enumerate(raw_data[1:]):
#     to_svg([to_frame(swipe)]).saveSvg(get_resource('2020-03-20_0 w%d.svg' % i))
#     break
