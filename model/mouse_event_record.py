from enum import Enum

from pydantic import BaseModel


class MouseEventType(Enum):
    click = 'click'
    mousemove = 'mousemove'
    mouseenter = 'mouseenter'
    mouseleave = 'mouseleave'
    mouseover = 'mouseover'
    mousedown = 'mousedown'
    mouseup = 'mouseup'


class MouseEventRecord(BaseModel):
    time: float
    x: float
    y: float
    type: MouseEventType
    element: str
