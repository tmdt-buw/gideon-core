from enum import Enum
from typing import List, Optional, Tuple, Dict
from uuid import UUID, uuid4 as uuid

from pydantic import BaseModel, Field

from model.mouse_event_record import MouseEventRecord


class Severity(Enum):
    okay = 'okay'
    warning = 'warning'
    error = 'error'


class LabelClass(BaseModel):
    name: str
    severity: Severity


class Label(BaseModel):
    label_class: LabelClass
    start: float
    end: float
    sample: int
    dimensions: List[int] = []


class AbstractLabel(BaseModel):
    label_type: int
    label_severity: Severity
    start: float
    end: float
    dimensions: List[int] = []


class PanZoom(BaseModel):
    x: int = 0
    y: int = 0
    zoom: float = 0.0


class AbstractPanZoom(BaseModel):
    x: float = 0.0
    y: float = 0.0
    zoom: float = 0.0


class Chart(BaseModel):
    dimensions: List[int] = []
    data: List[List[Tuple[str, float]]] = []
    labels: List[Label] = []
    panZoom: Optional[PanZoom]
    events: List[MouseEventRecord] = []


class AbstractChart(BaseModel):
    dimensions: List[int] = []
    data_distances: List[List[float]] = []
    labels: List[AbstractLabel] = []
    panZoom: Optional[AbstractPanZoom]


class State(BaseModel):
    id: UUID = Field(default_factory=uuid)
    charts: List[Chart] = []


class Action(BaseModel):
    type: str
    parameters: Optional[Dict]
    source: Optional[UUID]
    to: Optional[UUID]


class AbstractState(BaseModel):
    ref: UUID
    score: int
    active_time: int
    charts: List[AbstractChart] = []


class Session(BaseModel):
    states: List[State] = []
    actions: List[Action] = []


class AbstractSession(BaseModel):
    states: List[AbstractState] = []


class InteractionData(BaseModel):
    sessions: List[Session] = []


class AbstractInteractionData(BaseModel):
    sessions: List[AbstractSession] = []
