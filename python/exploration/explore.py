from keyboard._0_types import Input, Keyboard, SwipeDataFrame, SwipeEmbeddingDataFrame
from keyboard._1_import import keyboards, _2020_03_20_0
from utilities import concat, get_resource, is_list_of, read_all, skip, split_at, split_by, windowed_2
from typing import List, Union, Optional, Tuple  # noqa
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

    path = draw.Path(stroke_width=5, stroke='green', fill='transparent')
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

def to_frames(data: SwipeEmbeddingDataFrame) -> draw.Drawing:
    return to_frames2(data.swipes, data.words)

def to_frames2(swipes: List[SwipeDataFrame], words: Optional[List[str]] = None) -> draw.Drawing:

    frames = [to_frame(swipe) for swipe in swipes]

    if words is not None:
        def group_with_word(frame, word, i):
            group = draw.Group()
            group.append(frame)
            group.append(draw.Text(str(i) + ': ' + word, fontSize=40, x=10, y=-40, transform='scale(1, -1)'))
            return group

        frames = [group_with_word(*t) for t in zip(frames, words, range(max(len(frames), len(words))))]

    animate_from_frames(frames, 0.2)


    return to_svg(frames)

def save_as_individual_frames(swipes: List[SwipeDataFrame], path_format):
    assert '%s' in path_format

def duration_per_word(data: SwipeEmbeddingDataFrame) -> List[int]:
    def duration_of_word(swipe: SwipeDataFrame) -> int:
        return swipe['Timestamp'].max() - swipe['Timestamp'].min()

    return [duration_of_word(row.swipes) for row in data.rows()]

def duration_per_char(data: SwipeEmbeddingDataFrame) -> List[int]:
    """ Gets the average duration per char for each word. """
    durations = duration_per_word(data)
    return [duration // len(row.words) for row, duration in zip(data.rows(), durations)]

def get_average_duration_per_char(data) -> List[int]:
    durations = duration_per_char(data)
    return sum(durations) / len(durations)

def average_n_datapoints_per_char(data: SwipeEmbeddingDataFrame):
    return sum(len(row.swipes) for row in data.rows()) / sum(len(row.words) for row in data.rows())


if __name__ == "__main__":
    from keyboard._1a_generate import df
    svg = to_frames(df)
    svg.saveSvg(get_resource('generated_random_word.svg'))
    # svg = to_frames(*_2020_03_20_0()[1:3])
    # svg.saveSvg(get_resource('2020-03-20_0 all words.svg'))
