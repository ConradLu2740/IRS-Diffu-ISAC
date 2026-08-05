"""
setup_sat.py — 星-地 ISAC 动态场景物理层

基于真实 TLE（SGP4）的 LEO 卫星轨道传播，为"星-地 ISAC"仿真提供：
  - 动态几何：卫星 / 星载RIS / 地面RIS / 地面目标(ROI) / 地面站(UE) 每帧位置
  - 远场信道：km 级距离的自由空间传播（保留与 setup.py 一致的 H ~ exp(j2πd/λ)/d 形式）
  - 多普勒频移：由卫星运动产生的径向速度注入
  - 传播时延：τ = d / c

坐标系约定：
  - 轨道传播使用 ECI（地心惯性），SGP4 输出 km / km·s⁻¹
  - 场景几何统一使用 ECEF（地心地固），便于地面站/目标固定
  - 多普勒与信道在 ECEF 下计算（ECEF 速度含地球自转修正）

依赖：sgp4 (pip install sgp4)
"""

import numpy as np
from sgp4.api import Satrec, jday

# ----------------------------------------------------------------------
# 物理常数
# ----------------------------------------------------------------------
C_LIGHT_KM = 299792.458          # 光速 km/s
EARTH_OMEGA = 7.2921159e-5       # 地球自转角速度 rad/s
EARTH_R_KM = 6378.137            # 地球赤道半径 km（简化球体）

# ----------------------------------------------------------------------
# 仿真参数（星-地 ISAC 默认）
# ----------------------------------------------------------------------
FC_HZ = 30e9                     # 载频 30 GHz（毫米波，λ = 1 cm）
WAVELENGTH_M = C_LIGHT_KM * 1000 / FC_HZ   # 0.01 m
TAU = 8                          # 时间帧数（与 setup.py 一致）
FRAME_INTERVAL_S = 1.0           # 帧间隔（秒），8 帧覆盖 ~60 km 弧长
MIN_ELEVATION_DEG = 20.0         # 过境窗口的最低仰角（度）
SEARCH_HOURS = 48                # 过境窗口搜索时长（小时）

# ----------------------------------------------------------------------
# 真实 TLE 选项：ISS (NORAD 25544) 与 Starlink (NORAD 44714)
# 历元均约 2026-08-03（Celestrak 获取）
# ----------------------------------------------------------------------
ISS_TLE = [
    "ISS (ZARYA)             ",
    "1 25544U 98067A   26215.79638706  .00007444  00000+0  14146-3 0  9999",
    "2 25544  51.6316  64.4821 0007224   9.2337 350.8783 15.49332738579132",
]

STARLINK_TLE = [
    "STARLINK-1008           ",
    "1 44714U 19074B   26215.97871058  .00029734  00000+0  37585-3 0  9997",
    "2 44714  53.1486 198.6893 0006328  18.4929 341.6313 15.59291253371649",
]

# 场景地面位置（默认：目标区域与地面站位于中国东部附近）
TARGET_LAT_DEG, TARGET_LON_DEG = 30.0, 120.0     # 目标区域中心（ROI）
GROUND_LAT_DEG, GROUND_LON_DEG = 30.0, 119.5     # 地面站（UE）
GROUND_ALT_KM = 0.0

# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------

def load_satrec(tle_lines):
    """从两行根数创建 SGP4 卫星对象。"""
    sat = Satrec.twoline2rv(tle_lines[1], tle_lines[2])
    return sat


def jd_to_jd_fr(yr, mo, day, hr, mi, sec):
    """把 UTC 日期时间转为 sgp4 需要的 (jd, fr)。"""
    jd, fr = jday(yr, mo, day, hr, mi, sec)
    return jd, fr


def gmst_deg(jd_full):
    """格林尼治平均恒星时（度）。jd_full = jd + fr。"""
    T = (jd_full - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (jd_full - 2451545.0)
            + 0.000387933 * T * T - T * T * T / 38710000.0)
    return gmst % 360.0


def eci_to_ecef(r_eci, v_eci, jd_full):
    """ECI 位置/速度 (km, km/s) → ECEF（含地球自转速度修正）。"""
    th = np.radians(gmst_deg(jd_full))
    ct, st = np.cos(th), np.sin(th)
    R = np.array([[ct, st, 0.0], [-st, ct, 0.0], [0.0, 0.0, 1.0]])
    r_ecef = R @ np.asarray(r_eci, dtype=float)
    v_ecef = R @ np.asarray(v_eci, dtype=float) - np.cross(
        np.array([0.0, 0.0, EARTH_OMEGA]), r_ecef)
    return r_ecef, v_ecef


def propagate_ecef(sat, yr, mo, day, hr, mi, sec):
    """传播卫星到指定 UTC 时刻，返回 ECEF 位置(km)与速度(km/s)。"""
    jd, fr = jday(yr, mo, day, hr, mi, sec)
    err, r_eci, v_eci = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 propagation error code={err}")
    r_ecef, v_ecef = eci_to_ecef(r_eci, v_eci, jd + fr)
    return r_ecef, v_ecef, jd + fr


def geodetic_to_ecef(lat_deg, lon_deg, alt_km=0.0):
    """经纬高 → ECEF (km)。简化球体模型，科研仿真足够。"""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    r = EARTH_R_KM + alt_km
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ], dtype=float)


def elevation_deg(from_ecef, to_ecef):
    """从 from_ecef 位置看 to_ecef 的仰角（度）。"""
    r_from = np.asarray(from_ecef, dtype=float)
    r_to = np.asarray(to_ecef, dtype=float)
    up = r_from / np.linalg.norm(r_from)
    v = r_to - r_from
    sin_el = np.dot(v, up) / (np.linalg.norm(v) + 1e-12)
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))


def radial_velocity_mps(v_a_km_s, v_b_km_s, a_km, b_km):
    """链路 a→b 的径向相对速度（m/s），沿 a→b 视线方向为正。"""
    u = (np.asarray(b_km, dtype=float) - np.asarray(a_km, dtype=float))
    u = u / (np.linalg.norm(u) + 1e-12)
    return np.dot(np.asarray(v_b_km_s, dtype=float) - np.asarray(v_a_km_s, dtype=float), u) * 1000.0


# ----------------------------------------------------------------------
# 过境窗口搜索
# ----------------------------------------------------------------------

def find_overpass_windows(sat, target_ecef, start_utc, search_hours=SEARCH_HOURS,
                          min_elevation_deg=MIN_ELEVATION_DEG, step_s=10.0):
    """扫描未来一段时间，返回卫星相对目标仰角超过阈值的连续窗口列表。

    返回 [(t0_sec, t1_sec), ...]，t 为相对 start_utc 的秒数。
    """
    yr, mo, day, hr, mi, sec = start_utc
    jd0, fr0 = jday(yr, mo, day, hr, mi, sec)
    jd0_full = jd0 + fr0

    windows = []
    prev_above = False
    win_start = None
    n_steps = int(search_hours * 3600 / step_s)

    for i in range(n_steps + 1):
        t_sec = i * step_s
        jd_full = jd0_full + t_sec / 86400.0
        jd_i, fr_i = np.floor(jd_full), jd_full - np.floor(jd_full)
        err, r_eci, v_eci = sat.sgp4(int(jd_i), fr_i)
        if err != 0:
            continue
        r_ecef, _ = eci_to_ecef(r_eci, v_eci, jd_full)
        el = elevation_deg(target_ecef, r_ecef)
        above = el >= min_elevation_deg

        if above and not prev_above:
            win_start = t_sec
        elif not above and prev_above and win_start is not None:
            windows.append((win_start, t_sec))
            win_start = None
        prev_above = above

    if win_start is not None:
        windows.append((win_start, n_steps * step_s))
    return windows


# ----------------------------------------------------------------------
# 场景类
# ----------------------------------------------------------------------

class SatISACScenario:
    """星-地 ISAC 动态场景。

    通过场景类生成 Tau 帧的几何/信道/多普勒/时延数据。
    角色：
      - BS:   LEO 卫星（发射导频）
      - ROI:  地面目标区域（感知对象，点云来自目标模板）
      - UE:   地面站（接收）
      - RIS:  星载（挂在卫星上）或地面（在地面站旁），模式可选
    """

    def __init__(self, tle_lines=ISS_TLE, fc_hz=FC_HZ, tau=TAU,
                 frame_interval_s=FRAME_INTERVAL_S,
                 target_lat=TARGET_LAT_DEG, target_lon=TARGET_LON_DEG,
                 ground_lat=GROUND_LAT_DEG, ground_lon=GROUND_LON_DEG,
                 min_elevation_deg=MIN_ELEVATION_DEG,
                 search_hours=SEARCH_HOURS, start_utc=None, sat_name="ISS"):
        self.tle_lines = tle_lines
        self.sat_name = sat_name
        self.sat = load_satrec(tle_lines)
        self.fc_hz = fc_hz
        self.wavelength_m = C_LIGHT_KM * 1000 / fc_hz
        self.tau = tau
        self.frame_interval_s = frame_interval_s
        self.min_elevation_deg = min_elevation_deg
        self.search_hours = search_hours

        # 地面固定点
        self.target_ecef = geodetic_to_ecef(target_lat, target_lon, 0.0)
        self.ground_ecef = geodetic_to_ecef(ground_lat, ground_lon, GROUND_ALT_KM)
        self.target_lat, self.target_lon = target_lat, target_lon
        self.ground_lat, self.ground_lon = ground_lat, ground_lon

        # 默认场景起始时间：TLE 历元（保证传播精度）
        if start_utc is None:
            start_utc = (2026, 8, 5, 12, 0, 0)
        self.start_utc = start_utc

        # 星载 RIS 相对卫星的偏移（本体近似：沿卫星径向/轨道切向的刚体偏移, km）
        # 简化为 ECEF 中一个固定小偏移（几十米），后续可细化星体姿态
        self.sat_ris_offset_km = np.array([0.0, 0.05, 0.02])   # ~50m 量级
        # 地面 RIS 相对地面站的偏移 (km)
        self.ground_ris_offset_km = np.array([0.002, 0.0, 0.0])  # 2m 量级

    # ------------------------------------------------------------------
    def find_overpass(self):
        windows = find_overpass_windows(
            self.sat, self.target_ecef, self.start_utc,
            search_hours=self.search_hours,
            min_elevation_deg=self.min_elevation_deg)
        return windows

    # ------------------------------------------------------------------
    def build_frames(self, window_center_sec=None):
        """在过境窗口中心附近生成 tau 帧的动态几何。

        返回 dict，每帧包含：
          t_sec: 相对帧 0 的秒数（多普勒相位累积用，0, dt, 2dt, ...）
          t_abs_sec: 相对场景起点的绝对秒数
          sat_pos, sat_vel, sat_ris_pos, ground_ris_pos,
          target_pos, ground_pos, dist_sat_target, dist_target_ground,
          elevation_deg, f_d_hz, delay_s
        """
        windows = self.find_overpass()
        if not windows:
            raise RuntimeError("未找到过境窗口，请调整仰角阈值或起始时间。")
        win = windows[0]
        center = window_center_sec if window_center_sec is not None else (win[0] + win[1]) / 2.0

        jd0, fr0 = jday(*self.start_utc)
        jd0_full = jd0 + fr0

        frames = []
        t_abs_start = center - (self.tau - 1) * self.frame_interval_s / 2.0
        for i in range(self.tau):
            t_sec = i * self.frame_interval_s          # 相对帧 0（相位累积用）
            t_abs_sec = t_abs_start + i * self.frame_interval_s
            jd_full = jd0_full + t_abs_sec / 86400.0
            jd_i, fr_i = np.floor(jd_full), jd_full - np.floor(jd_full)
            err, r_eci, v_eci = self.sat.sgp4(int(jd_i), fr_i)
            if err != 0:
                raise RuntimeError(f"SGP4 error at frame {i}: code={err}")
            sat_pos, sat_vel = eci_to_ecef(r_eci, v_eci, jd_full)

            # 星载 RIS：卫星 + 刚体偏移（简化，不随姿态旋转）
            sat_ris_pos = sat_pos + self.sat_ris_offset_km
            ground_ris_pos = self.ground_ecef + self.ground_ris_offset_km

            # 关键链路距离
            d_st = np.linalg.norm(sat_pos - self.target_ecef)        # 卫星→目标
            d_tg = np.linalg.norm(self.target_ecef - self.ground_ecef)  # 目标→地面站

            # 多普勒：感知路径 BS→ROI（发射端卫星运动主导）
            # f_d = (v_rel · u) / λ，u 为 卫星→目标 方向
            v_rel = radial_velocity_mps(sat_vel, np.zeros(3), sat_pos, self.target_ecef)
            f_d = v_rel / self.wavelength_m   # Hz

            # 时延：BS→ROI→UE 总路径
            delay = (d_st + d_tg) / C_LIGHT_KM  # s

            el = elevation_deg(self.target_ecef, sat_pos)

            frames.append({
                "t_sec": t_sec,
                "t_abs_sec": t_abs_sec,
                "sat_pos": sat_pos, "sat_vel": sat_vel,
                "sat_ris_pos": sat_ris_pos,
                "ground_ris_pos": ground_ris_pos,
                "target_pos": self.target_ecef.copy(),
                "ground_pos": self.ground_ecef.copy(),
                "dist_sat_target": d_st,
                "dist_target_ground": d_tg,
                "elevation_deg": el,
                "f_d_hz": f_d,
                "delay_s": delay,
            })
        return frames

    # ------------------------------------------------------------------
    def get_channel_far(self, a_km, b_km, t_sec, v_a_km_s=None, v_b_km_s=None):
        """远场自由空间信道（与 setup.py 的 get_Channel 形式一致，距离用米）。

        H = sqrt(0.1) * exp(j 2π d/λ) / d   （d 单位 m）
        返回 (H, dist_km, f_d_hz, delay_s)
        """
        a = np.asarray(a_km, dtype=float)
        b = np.asarray(b_km, dtype=float)
        d_km = np.linalg.norm(b - a)
        d_m = d_km * 1000.0

        phase = 2.0 * np.pi * d_m / self.wavelength_m
        H = np.sqrt(0.1) * np.exp(1j * phase) / max(d_m, 1e-6)

        # 多普勒：若给出链路两端速度则计算径向分量
        f_d = 0.0
        if v_a_km_s is not None and v_b_km_s is not None:
            v_rel = radial_velocity_mps(v_a_km_s, v_b_km_s, a, b)
            f_d = v_rel / self.wavelength_m
            # t_sec 为相对参考时刻的秒数（帧 0），相位累积 e^{j2π f_d t}
            H = H * np.exp(1j * 2.0 * np.pi * f_d * t_sec)

        delay = d_km / C_LIGHT_KM
        return H, d_km, f_d, delay


# ----------------------------------------------------------------------
# 便捷入口
# ----------------------------------------------------------------------

def build_default_scenario():
    """创建默认星-地 ISAC 场景并生成帧数据，返回 (scenario, frames)。"""
    scenario = SatISACScenario()
    frames = scenario.build_frames()
    return scenario, frames


if __name__ == "__main__":
    scenario, frames = build_default_scenario()
    print(f"默认场景：载频 {scenario.fc_hz/1e9:.1f} GHz (λ={scenario.wavelength_m*100:.1f} cm)")
    print(f"目标区域: ({scenario.target_lat}N, {scenario.target_lon}E)  地面站: ({scenario.ground_lat}N, {scenario.ground_lon}E)")
    print(f"过境窗口数(≥{scenario.min_elevation_deg}°): {len(scenario.find_overpass())}")
    print(f"\n{'帧':>3} {'仰角°':>8} {'卫星-目标km':>12} {'目标-地面km':>12} {'多普勒kHz':>10} {'时延ms':>8}")
    for i, f in enumerate(frames):
        print(f"{i:3d} {f['elevation_deg']:8.2f} {f['dist_sat_target']:12.2f} {f['dist_target_ground']:12.2f} "
              f"{f['f_d_hz']/1e3:10.2f} {f['delay_s']*1e3:8.2f}")
