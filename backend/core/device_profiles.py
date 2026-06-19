# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Target device profiles for deployment optimization recommendations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    """Hardware profile for a target deployment device."""
    name: str
    category: str           # "iphone" | "ipad" | "mac"
    ram_gb: float            # Total system RAM
    available_ram_gb: float  # Realistic available RAM for model
    neural_engine_tops: float  # Neural Engine TOPS (0 if none)
    gpu_cores: int
    chip: str

    @property
    def max_model_size_gb(self) -> float:
        """Conservative max model size (leave room for runtime overhead)."""
        return self.available_ram_gb * 0.85


# Apple Silicon device database
DEVICE_PROFILES: dict[str, DeviceProfile] = {
    # ---------- iPhones ----------
    "iPhone 15 Pro": DeviceProfile(
        name="iPhone 15 Pro", category="iphone",
        ram_gb=8, available_ram_gb=4.5,
        neural_engine_tops=35, gpu_cores=6, chip="A17 Pro",
    ),
    "iPhone 16 Pro": DeviceProfile(
        name="iPhone 16 Pro", category="iphone",
        ram_gb=8, available_ram_gb=4.5,
        neural_engine_tops=38, gpu_cores=6, chip="A18 Pro",
    ),
    "iPhone 16 Pro Max": DeviceProfile(
        name="iPhone 16 Pro Max", category="iphone",
        ram_gb=12, available_ram_gb=6,
        neural_engine_tops=38, gpu_cores=6, chip="A18 Pro",
    ),
    "iPhone Air": DeviceProfile(
        name="iPhone Air", category="iphone",
        ram_gb=12, available_ram_gb=6,
        neural_engine_tops=38, gpu_cores=5, chip="A19",
    ),
    "iPhone 17 Pro": DeviceProfile(
        name="iPhone 17 Pro", category="iphone",
        ram_gb=12, available_ram_gb=6,
        neural_engine_tops=38, gpu_cores=6, chip="A19 Pro",
    ),
    "iPhone 17 Pro Max": DeviceProfile(
        name="iPhone 17 Pro Max", category="iphone",
        ram_gb=12, available_ram_gb=6,
        neural_engine_tops=38, gpu_cores=6, chip="A19 Pro",
    ),
    "iPhone 17e": DeviceProfile(
        name="iPhone 17e", category="iphone",
        ram_gb=8, available_ram_gb=6,
        neural_engine_tops=38, gpu_cores=5, chip="A19",
    ),

    # ---------- iPads ----------
    "iPad Air M2": DeviceProfile(
        name="iPad Air M2", category="ipad",
        ram_gb=8, available_ram_gb=5.5,
        neural_engine_tops=15.8, gpu_cores=10, chip="M2",
    ),
    "iPad Air M3": DeviceProfile(
        name="iPad Air M3", category="ipad",
        ram_gb=8, available_ram_gb=5.5,
        neural_engine_tops=18, gpu_cores=10, chip="M3",
    ),
    "iPad Pro M4": DeviceProfile(
        name="iPad Pro M4", category="ipad",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=38, gpu_cores=10, chip="M4",
    ),
    "iPad Pro M4 (8GB)": DeviceProfile(
        name="iPad Pro M4 (8GB)", category="ipad",
        ram_gb=8, available_ram_gb=5.5,
        neural_engine_tops=38, gpu_cores=10, chip="M4",
    ),
    "iPad Pro M5": DeviceProfile(
        name="iPad Pro M5", category="ipad",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),
    "iPad Pro M5 (12GB)": DeviceProfile(
        name="iPad Pro M5 (12GB)", category="ipad",
        ram_gb=12, available_ram_gb=8.5,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),
    "iPad Pro M5 (16GB)": DeviceProfile(
        name="iPad Pro M5 (16GB)", category="ipad",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),

    # ---------- Macs — M1 generation ----------
    "MacBook Air M1 (8GB)": DeviceProfile(
        name="MacBook Air M1 (8GB)", category="mac",
        ram_gb=8, available_ram_gb=5.5,
        neural_engine_tops=11, gpu_cores=8, chip="M1",
    ),
    "MacBook Air M1 (16GB)": DeviceProfile(
        name="MacBook Air M1 (16GB)", category="mac",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=11, gpu_cores=8, chip="M1",
    ),
    "MacBook Pro M1 Pro (16GB)": DeviceProfile(
        name="MacBook Pro M1 Pro (16GB)", category="mac",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=11, gpu_cores=16, chip="M1 Pro",
    ),
    "MacBook Pro M1 Pro (32GB)": DeviceProfile(
        name="MacBook Pro M1 Pro (32GB)", category="mac",
        ram_gb=32, available_ram_gb=26,
        neural_engine_tops=11, gpu_cores=16, chip="M1 Pro",
    ),
    "MacBook Pro M1 Max (32GB)": DeviceProfile(
        name="MacBook Pro M1 Max (32GB)", category="mac",
        ram_gb=32, available_ram_gb=26,
        neural_engine_tops=11, gpu_cores=32, chip="M1 Max",
    ),
    "MacBook Pro M1 Max (64GB)": DeviceProfile(
        name="MacBook Pro M1 Max (64GB)", category="mac",
        ram_gb=64, available_ram_gb=54,
        neural_engine_tops=11, gpu_cores=32, chip="M1 Max",
    ),
    "Mac Studio M1 Ultra (64GB)": DeviceProfile(
        name="Mac Studio M1 Ultra (64GB)", category="mac",
        ram_gb=64, available_ram_gb=54,
        neural_engine_tops=22, gpu_cores=48, chip="M1 Ultra",
    ),
    "Mac Studio M1 Ultra (128GB)": DeviceProfile(
        name="Mac Studio M1 Ultra (128GB)", category="mac",
        ram_gb=128, available_ram_gb=110,
        neural_engine_tops=22, gpu_cores=64, chip="M1 Ultra",
    ),

    # ---------- Macs — M2 generation ----------
    "MacBook Air M2 (8GB)": DeviceProfile(
        name="MacBook Air M2 (8GB)", category="mac",
        ram_gb=8, available_ram_gb=5.5,
        neural_engine_tops=15.8, gpu_cores=10, chip="M2",
    ),
    "MacBook Air M2 (16GB)": DeviceProfile(
        name="MacBook Air M2 (16GB)", category="mac",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=15.8, gpu_cores=10, chip="M2",
    ),
    "MacBook Air M2 (24GB)": DeviceProfile(
        name="MacBook Air M2 (24GB)", category="mac",
        ram_gb=24, available_ram_gb=18,
        neural_engine_tops=15.8, gpu_cores=10, chip="M2",
    ),
    "MacBook Pro M2 Pro (16GB)": DeviceProfile(
        name="MacBook Pro M2 Pro (16GB)", category="mac",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=15.8, gpu_cores=19, chip="M2 Pro",
    ),
    "MacBook Pro M2 Pro (32GB)": DeviceProfile(
        name="MacBook Pro M2 Pro (32GB)", category="mac",
        ram_gb=32, available_ram_gb=26,
        neural_engine_tops=15.8, gpu_cores=19, chip="M2 Pro",
    ),
    "MacBook Pro M2 Max (32GB)": DeviceProfile(
        name="MacBook Pro M2 Max (32GB)", category="mac",
        ram_gb=32, available_ram_gb=26,
        neural_engine_tops=15.8, gpu_cores=38, chip="M2 Max",
    ),
    "MacBook Pro M2 Max (64GB)": DeviceProfile(
        name="MacBook Pro M2 Max (64GB)", category="mac",
        ram_gb=64, available_ram_gb=54,
        neural_engine_tops=15.8, gpu_cores=38, chip="M2 Max",
    ),
    "Mac Studio M2 Ultra (64GB)": DeviceProfile(
        name="Mac Studio M2 Ultra (64GB)", category="mac",
        ram_gb=64, available_ram_gb=54,
        neural_engine_tops=31.6, gpu_cores=60, chip="M2 Ultra",
    ),
    "Mac Studio M2 Ultra (128GB)": DeviceProfile(
        name="Mac Studio M2 Ultra (128GB)", category="mac",
        ram_gb=128, available_ram_gb=110,
        neural_engine_tops=31.6, gpu_cores=76, chip="M2 Ultra",
    ),
    "Mac Studio M2 Ultra (192GB)": DeviceProfile(
        name="Mac Studio M2 Ultra (192GB)", category="mac",
        ram_gb=192, available_ram_gb=165,
        neural_engine_tops=31.6, gpu_cores=76, chip="M2 Ultra",
    ),

    # ---------- Macs — M3 generation ----------
    "MacBook Air M3 (8GB)": DeviceProfile(
        name="MacBook Air M3 (8GB)", category="mac",
        ram_gb=8, available_ram_gb=5.5,
        neural_engine_tops=18, gpu_cores=10, chip="M3",
    ),
    "MacBook Air M3 (16GB)": DeviceProfile(
        name="MacBook Air M3 (16GB)", category="mac",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=18, gpu_cores=10, chip="M3",
    ),
    "MacBook Air M3 (24GB)": DeviceProfile(
        name="MacBook Air M3 (24GB)", category="mac",
        ram_gb=24, available_ram_gb=18,
        neural_engine_tops=18, gpu_cores=10, chip="M3",
    ),
    "MacBook Pro M3 Pro (18GB)": DeviceProfile(
        name="MacBook Pro M3 Pro (18GB)", category="mac",
        ram_gb=18, available_ram_gb=14,
        neural_engine_tops=18, gpu_cores=14, chip="M3 Pro",
    ),
    "MacBook Pro M3 Pro (36GB)": DeviceProfile(
        name="MacBook Pro M3 Pro (36GB)", category="mac",
        ram_gb=36, available_ram_gb=28,
        neural_engine_tops=18, gpu_cores=14, chip="M3 Pro",
    ),
    "MacBook Pro M3 Max (36GB)": DeviceProfile(
        name="MacBook Pro M3 Max (36GB)", category="mac",
        ram_gb=36, available_ram_gb=28,
        neural_engine_tops=18, gpu_cores=30, chip="M3 Max",
    ),
    "MacBook Pro M3 Max (48GB)": DeviceProfile(
        name="MacBook Pro M3 Max (48GB)", category="mac",
        ram_gb=48, available_ram_gb=38,
        neural_engine_tops=18, gpu_cores=40, chip="M3 Max",
    ),
    "MacBook Pro M3 Max (96GB)": DeviceProfile(
        name="MacBook Pro M3 Max (96GB)", category="mac",
        ram_gb=96, available_ram_gb=80,
        neural_engine_tops=18, gpu_cores=40, chip="M3 Max",
    ),
    "MacBook Pro M3 Max (128GB)": DeviceProfile(
        name="MacBook Pro M3 Max (128GB)", category="mac",
        ram_gb=128, available_ram_gb=110,
        neural_engine_tops=18, gpu_cores=40, chip="M3 Max",
    ),

    # ---------- Macs — M4 generation ----------
    "MacBook Air M4 (16GB)": DeviceProfile(
        name="MacBook Air M4 (16GB)", category="mac",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=38, gpu_cores=10, chip="M4",
    ),
    "MacBook Air M4 (24GB)": DeviceProfile(
        name="MacBook Air M4 (24GB)", category="mac",
        ram_gb=24, available_ram_gb=18,
        neural_engine_tops=38, gpu_cores=10, chip="M4",
    ),
    "MacBook Air M4 (32GB)": DeviceProfile(
        name="MacBook Air M4 (32GB)", category="mac",
        ram_gb=32, available_ram_gb=26,
        neural_engine_tops=38, gpu_cores=10, chip="M4",
    ),
    "MacBook Pro M4 Pro (24GB)": DeviceProfile(
        name="MacBook Pro M4 Pro (24GB)", category="mac",
        ram_gb=24, available_ram_gb=18,
        neural_engine_tops=38, gpu_cores=16, chip="M4 Pro",
    ),
    "MacBook Pro M4 Pro (48GB)": DeviceProfile(
        name="MacBook Pro M4 Pro (48GB)", category="mac",
        ram_gb=48, available_ram_gb=38,
        neural_engine_tops=38, gpu_cores=20, chip="M4 Pro",
    ),
    "MacBook Pro M4 Max (36GB)": DeviceProfile(
        name="MacBook Pro M4 Max (36GB)", category="mac",
        ram_gb=36, available_ram_gb=28,
        neural_engine_tops=38, gpu_cores=32, chip="M4 Max",
    ),
    "MacBook Pro M4 Max (48GB)": DeviceProfile(
        name="MacBook Pro M4 Max (48GB)", category="mac",
        ram_gb=48, available_ram_gb=38,
        neural_engine_tops=38, gpu_cores=40, chip="M4 Max",
    ),
    "MacBook Pro M4 Max (64GB)": DeviceProfile(
        name="MacBook Pro M4 Max (64GB)", category="mac",
        ram_gb=64, available_ram_gb=54,
        neural_engine_tops=38, gpu_cores=40, chip="M4 Max",
    ),
    "MacBook Pro M4 Max (128GB)": DeviceProfile(
        name="MacBook Pro M4 Max (128GB)", category="mac",
        ram_gb=128, available_ram_gb=110,
        neural_engine_tops=38, gpu_cores=40, chip="M4 Max",
    ),
    "MacBook Air M5 (16GB)": DeviceProfile(
        name="MacBook Air M5 (16GB)", category="mac",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),
    "MacBook Air M5 (24GB)": DeviceProfile(
        name="MacBook Air M5 (24GB)", category="mac",
        ram_gb=24, available_ram_gb=18,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),
    "MacBook Air M5 (32GB)": DeviceProfile(
        name="MacBook Air M5 (32GB)", category="mac",
        ram_gb=32, available_ram_gb=26,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),
    "MacBook Pro M5 (16GB)": DeviceProfile(
        name="MacBook Pro M5 (16GB)", category="mac",
        ram_gb=16, available_ram_gb=12,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),
    "MacBook Pro M5 (24GB)": DeviceProfile(
        name="MacBook Pro M5 (24GB)", category="mac",
        ram_gb=24, available_ram_gb=18,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),
    "MacBook Pro M5 (32GB)": DeviceProfile(
        name="MacBook Pro M5 (32GB)", category="mac",
        ram_gb=32, available_ram_gb=26,
        neural_engine_tops=38, gpu_cores=10, chip="M5",
    ),
    "MacBook Pro M5 Pro (24GB)": DeviceProfile(
        name="MacBook Pro M5 Pro (24GB)", category="mac",
        ram_gb=24, available_ram_gb=18,
        neural_engine_tops=38, gpu_cores=20, chip="M5 Pro",
    ),
    "MacBook Pro M5 Pro (48GB)": DeviceProfile(
        name="MacBook Pro M5 Pro (48GB)", category="mac",
        ram_gb=48, available_ram_gb=38,
        neural_engine_tops=38, gpu_cores=20, chip="M5 Pro",
    ),
    "MacBook Pro M5 Pro (64GB)": DeviceProfile(
        name="MacBook Pro M5 Pro (64GB)", category="mac",
        ram_gb=64, available_ram_gb=54,
        neural_engine_tops=38, gpu_cores=20, chip="M5 Pro",
    ),
    "MacBook Pro M5 Max (36GB)": DeviceProfile(
        name="MacBook Pro M5 Max (36GB)", category="mac",
        ram_gb=36, available_ram_gb=28,
        neural_engine_tops=38, gpu_cores=32, chip="M5 Max",
    ),
    "MacBook Pro M5 Max (48GB)": DeviceProfile(
        name="MacBook Pro M5 Max (48GB)", category="mac",
        ram_gb=48, available_ram_gb=38,
        neural_engine_tops=38, gpu_cores=40, chip="M5 Max",
    ),
    "MacBook Pro M5 Max (64GB)": DeviceProfile(
        name="MacBook Pro M5 Max (64GB)", category="mac",
        ram_gb=64, available_ram_gb=54,
        neural_engine_tops=38, gpu_cores=40, chip="M5 Max",
    ),
    "MacBook Pro M5 Max (128GB)": DeviceProfile(
        name="MacBook Pro M5 Max (128GB)", category="mac",
        ram_gb=128, available_ram_gb=110,
        neural_engine_tops=38, gpu_cores=40, chip="M5 Max",
    ),
    "Mac Studio M4 Max (36GB)": DeviceProfile(
        name="Mac Studio M4 Max (36GB)", category="mac",
        ram_gb=36, available_ram_gb=28,
        neural_engine_tops=38, gpu_cores=32, chip="M4 Max",
    ),
    "Mac Studio M4 Max (48GB)": DeviceProfile(
        name="Mac Studio M4 Max (48GB)", category="mac",
        ram_gb=48, available_ram_gb=38,
        neural_engine_tops=38, gpu_cores=40, chip="M4 Max",
    ),
    "Mac Studio M4 Max (64GB)": DeviceProfile(
        name="Mac Studio M4 Max (64GB)", category="mac",
        ram_gb=64, available_ram_gb=54,
        neural_engine_tops=38, gpu_cores=40, chip="M4 Max",
    ),
    "Mac Studio M4 Max (128GB)": DeviceProfile(
        name="Mac Studio M4 Max (128GB)", category="mac",
        ram_gb=128, available_ram_gb=110,
        neural_engine_tops=38, gpu_cores=40, chip="M4 Max",
    ),
    "Mac Studio M3 Ultra (96GB)": DeviceProfile(
        name="Mac Studio M3 Ultra (96GB)", category="mac",
        ram_gb=96, available_ram_gb=80,
        neural_engine_tops=36, gpu_cores=60, chip="M3 Ultra",
    ),
    "Mac Studio M3 Ultra (256GB)": DeviceProfile(
        name="Mac Studio M3 Ultra (256GB)", category="mac",
        ram_gb=256, available_ram_gb=220,
        neural_engine_tops=36, gpu_cores=80, chip="M3 Ultra",
    ),
}


def get_device(name: str) -> DeviceProfile | None:
    return DEVICE_PROFILES.get(name)


def devices_by_category(category: str) -> list[DeviceProfile]:
    return [d for d in DEVICE_PROFILES.values() if d.category == category]


def all_devices() -> list[DeviceProfile]:
    return list(DEVICE_PROFILES.values())


def make_profile_from_ram(chip: str, ram_gb: float, category: str = "mac") -> DeviceProfile:
    """Create a synthetic DeviceProfile when no exact match exists.

    Uses conservative heuristics:
    - Mac: available = ram * 0.85 (OS + apps overhead ~15%)
    - iPhone/iPad: use Jetsam-aware estimates
    """
    if category == "iphone":
        available = min(ram_gb * 0.55, 6.5)  # Jetsam limit
    elif category == "ipad":
        available = min(ram_gb * 0.7, 12)
    else:
        available = ram_gb * 0.85

    return DeviceProfile(
        name=f"{chip} ({int(ram_gb)}GB)",
        category=category,
        ram_gb=ram_gb,
        available_ram_gb=round(available, 1),
        neural_engine_tops=0,
        gpu_cores=0,
        chip=chip,
    )


def recommend_target_size_gb(device: DeviceProfile) -> float:
    """Recommend target model size for a device."""
    return device.max_model_size_gb
