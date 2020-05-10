from typing import Union
import drawSvg as draw
from keyboard._0_types import Keyboard, Key
from keyboard._1_import import keyboards

def draw_keyboard(keyboard_or_index: Union[int, Keyboard]) -> draw.Group:
    keyboard: Keyboard = keyboards[keyboard_or_index] if isinstance(keyboard_or_index, int) else keyboard_or_index

    outlines = draw.Group(transform=f"translate({-keyboard.left},{-keyboard.top})")
    chars = draw.Group(transform="scale(1,-1)")

    for key in keyboard.values():
        outlines.append(draw.Rectangle(key.x, key.y, key.width, key.height, fill='transparent', stroke_width=5, stroke='blue'))
        if key.code > 0:
            chars.append(draw.Text(key.char, fontSize=25, x=key.x + key.width / 2, y=-key.y - key.height / 2, center=True))

    outlines.append(chars)
    return outlines
