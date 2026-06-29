"""
ripple_pulse.py

투명 액체(물 등)는 정지 이미지에서 반사·굴절 차이가 거의 없어
일반적인 Classification 모델로는 감지가 거의 불가능하다.

이를 해결하기 위해 회전판을 짧은 거리로 cw -> ccw 반복 구동시켜
컵 내부 액면에 인위적인 파동(ripple)을 만들고, 그 구간만 별도로
분리 수집하여 판별하는 방식을 직접 설계했다.

본 스니펫은 실제 운영 코드에서 모터 ID, 통신 정보, 사내 설정값을
모두 제거하고 핵심 로직만 일반화한 버전이다.
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class RippleConfig:
    motor_id: int
    forward_direction: str = "cw"
    back_direction: str = "ccw"
    distance: int = 1000          # 왕복 이동 거리 (장비 스펙에 맞게 조정)
    speed: int = 120
    accel: int = 80
    decel: int = 80
    settle_seconds: float = 0.10  # 정지->ripple 전환 텀


class MotorController:
    """실제 모터 드라이버 연동 인터페이스 (구현은 환경에 맞게 대체)."""

    async def run_and_wait(self, motor_id: int, direction: str,
                            distance: int, speed: int,
                            accel: int, decel: int) -> None:
        raise NotImplementedError


class FrameCollector:
    """카메라로부터 N개 프레임을 비동기로 수집하는 인터페이스."""

    async def collect_n(self, n: int) -> list[dict]:
        raise NotImplementedError

    async def collect_until(self, stop_event: asyncio.Event) -> list[dict]:
        raise NotImplementedError


class RippleVisionCollector:
    """
    정지 구간(B)과 ripple 구간(A)을 분리 수집하는 컨트롤러.

    - B(정지): 이물질·뚜껑처럼 정적인 특징은 흔들리지 않은 상태에서
      먼저 안정적으로 판별한다.
    - A(ripple): 액상류(휘핑크림/물)처럼 정지 상태에서 구분이 안 되는
      항목을 회전판 파동으로 가시화한 뒤 판별한다.
    """

    def __init__(self, motor: MotorController, collector: FrameCollector,
                 config: RippleConfig):
        self.motor = motor
        self.collector = collector
        self.config = config

    async def _ripple_pulse(self) -> None:
        """1회 cw -> ccw 왕복으로 파동 1회 생성."""
        cfg = self.config
        await self.motor.run_and_wait(
            cfg.motor_id, cfg.forward_direction, cfg.distance,
            cfg.speed, cfg.accel, cfg.decel,
        )
        await self.motor.run_and_wait(
            cfg.motor_id, cfg.back_direction, cfg.distance,
            cfg.speed, cfg.accel, cfg.decel,
        )

    async def _ripple_until_stop(self, stop_event: asyncio.Event) -> None:
        """목표 프레임 수에 도달할 때까지 파동을 반복 생성."""
        while not stop_event.is_set():
            await self._ripple_pulse()

    async def collect(self, frame_count_b: int, frame_count_a: int) -> dict:
        """
        B(정지) 먼저 수집 -> 짧은 settle 대기 -> A(ripple) 나중 수집.

        B를 먼저 수집하는 이유: 정지 상태가 가장 안정적인 기준값이므로
        컵 종류·이물질·뚜껑처럼 노이즈에 민감한 항목을 먼저 확정한다.
        """
        static_frames = await self.collector.collect_n(frame_count_b)

        if self.config.settle_seconds > 0:
            await asyncio.sleep(self.config.settle_seconds)

        stop_event = asyncio.Event()

        async def collect_and_stop() -> list[dict]:
            frames = []
            async for frame in self._iter_frames():
                frames.append(frame)
                if len(frames) >= frame_count_a:
                    stop_event.set()
                    break
            return frames

        ripple_frames, _ = await asyncio.gather(
            collect_and_stop(),
            self._ripple_until_stop(stop_event),
        )

        return {
            "static_frames": static_frames,
            "ripple_frames": ripple_frames,
        }

    async def _iter_frames(self):
        """실제 환경에서는 카메라 스트림 제너레이터로 대체."""
        raise NotImplementedError
