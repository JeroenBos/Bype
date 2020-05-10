from keyboard._0_types import Input, SwipeDataFrame
from keyboard._1_import import keyboards, _2020_03_20_0
from utilities import concat, get_resource, is_list_of, read_all, skip, split_at, split_by, windowed_2
from typing import List, Union, Optional  # noqa
from exploration.draw_keyboard import draw_keyboard
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


def to_frame(swipe: SwipeDataFrame):
    if len(swipe.X) == 0:
        raise ValueError('empty swipe')

    path = draw.Path(stroke_width=20, stroke='green', fill='transparent')
    path.M(swipe.X.iloc[0], swipe.Y.iloc[0])
    for x, y in skip(zip(swipe.X, swipe.Y), 1):
        path.L(x, y)
    return path


def to_svg(elements):
    keyboard_index = 0
    keyboard = keyboards[keyboard_index]
    width, height = keyboard.left + keyboard.width, keyboard.top + keyboard.height

    root = draw.Group(transform=f'scale(1,-1)')
    root.append(draw_keyboard(keyboard_index))
    root.extend(elements)
    svg = draw.Drawing(width, height, origin=(0, -height))
    svg.append(root)
    return svg


def to_frames(swipes: List[SwipeDataFrame], words: Optional[List[str]] = None):

    frames = [to_frame(swipe) for swipe in swipes]

    if words is not None:
        def group_with_word(frame, word):
            group = draw.Group()
            group.append(frame)
            group.append(draw.Text(word, fontSize=40, x=10, y=-40, transform='scale(1, -1)'))
            return group

        frames = [group_with_word(*t) for t in zip(frames, words)]

    animate_from_frames(frames, 0.2)


    return to_svg(frames)

def save_as_individual_frames(swipes: List[SwipeDataFrame], path_format):
    assert '%s' in path_format


empty_frame = draw.Group()
# words = text_file_to_swipes(raw_data
def correct(frames, words, frames_to_skip: List[int], extra_frames: List[int], words_to_merge: List[List[int]]):
    for combi in words_to_merge:
        words[combi[0]] = "".join(words[c] for c in combi)
        for r in sorted(combi[1:], reverse=True):
            del words[r]

    words_iter = iter(words)
    word_index = -1

    def next_word():
        nonlocal word_index
        word_index += 1
        try:
            return str(word_index) + ': ' + next(words_iter)
        except StopIteration:
            return str(word_index) + ': <no word>'


    for i, frame in enumerate(frames):
        if i in frames_to_skip:
            continue
        if i in extra_frames:
            yield empty_frame, next_word()

        yield frame, next_word()

def correct_and_skip(skip_count, *args):
    return skip(correct(*args), skip_count)


svg = to_frames(*_2020_03_20_0()[1:3])
svg.saveSvg(get_resource('2020-03-20_0 all words.svg'))


# for i, swipe in enumerate(raw_data[1:]):
#     to_svg([to_frame(swipe)]).saveSvg(get_resource('2020-03-20_0 w%d.svg' % i))
#     break
