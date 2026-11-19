import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.config import Config
from src.data.preprocess import segment_trajectory
from src.utils.geo import haversine


def _extract_single_ship_feature(args):
    if isinstance(args[0], tuple):
        (shipno, month_split, split_type), group = args
    else:
        shipno, group = args
        month_split, split_type = 0, '1month'
    
    group = group.sort_values('postime')
    
    # 初始化特征字典
    result = {
        'shipno': shipno,
        'month_split': month_split,
        'split_type': split_type,
        'sog_mean': 0, 'sog_std': 0, 'sog_max': 0, 'sog_min': 0, 'sog_median': 0,
        'cog_mean': 0, 'cog_std': 0, 'cog_max': 0, 'cog_min': 0, 'cog_median': 0,
        'point_count': 0, 'total_distance': 0, 'avg_speed': 0,
        'sog_change_rate': 0, 'cog_change_rate': 0, 'avg_time_interval': 0,
        'segment_count': 0, 'segment_sog_var': 0, 'segment_cog_var': 0,
        'distance_segment_sog_var': 0, 'distance_segment_cog_var': 0,
        'lat_range': 0, 'lon_range': 0,
        'curvature_mean': 0, 'curvature_std': 0,
        'turn_frequency': 0,
        'dwell_point_ratio': 0, 'dwell_region_count': 0,
        'month_duration': 0,
        'shape_complexity_lowfreq': 0, 'shape_complexity_highfreq': 0,
        'cluster_ratio': 0, 'cluster_count': 0,
        'directional_consistency': 0,
        'convex_hull_area': 0,
        'directional_entropy': 0,
        'high_speed_ratio': 0,
        'sog_smoothness': 0,
        'sog_entropy': 0,
        'dwell_event_duration_mean': 0,
        'dwell_event_frequency': 0,
        'dwell_spatial_spread': 0,
        'path_straightness': 0,
        'grid_repetition': 0,
        'dominant_cog_ratio': 0,
        'trajectory_compactness': 0,
        'spatial_autocorrelation': 0,
        'fractal_dimension': 0,
        'fft_sog_power_low': 0, 'fft_sog_power_high': 0,
        'fft_cog_power_low': 0, 'fft_cog_power_high': 0,
        'sog_acceleration_mean': 0, 'sog_acceleration_std': 0,
        'cog_change_entropy': 0,
        'grid_transition_freq': 0,
        'straight_segment_ratio': 0,
        'loop_segment_ratio': 0,
        'sog_skewness': 0,
        'sog_kurtosis': 0,
        'cog_skewness': 0,
        'cog_kurtosis': 0,
        'time_interval_entropy': 0,
        'max_time_interval': 0,
        'sog_autocorrelation': 0,
        'cog_autocorrelation': 0,
        'trajectory_crossings': 0,
        'grid_density': 0,
        'grid_entropy': 0,
        'centroid_shift': 0,
        'acceleration_event_count': 0,
        'acceleration_event_duration_mean': 0,
        'sharp_turn_count': 0,
        'straight_segment_length_mean': 0,
        'dwell_point_spatial_mean': 0,
        'fft_sog_power_mid': 0,
        'fft_cog_power_mid': 0,
        'sog_cog_correlation': 0,
        'grid_transition_speed': 0,
        'kmeans_cluster_density': 0,
        'shiptype': group['shiptype'].iloc[0] if 'shiptype' in group.columns and not group.empty else None
    }
    
    if len(group) < 2:
        return result
    
    # 提取基本字段
    sog = np.array(group['sog'], dtype=np.float32)
    cog = np.array(group['cog'], dtype=np.float32)
    lat = np.array(group['lat'], dtype=np.float32)
    lon = np.array(group['lon'], dtype=np.float32)
    postime = group['postime']
    point_count = len(group)
    
    # 基本统计特征
    sog_stats = {
        'mean': float(np.mean(sog)),
        'std': float(np.std(sog)),
        'max': float(np.max(sog)),
        'min': float(np.min(sog)),
        'median': float(np.median(sog)),
        'skewness': float(stats.skew(sog)) if len(sog) > 2 else 0,
        'kurtosis': float(stats.kurtosis(sog)) if len(sog) > 2 else 0
    }
    cog_stats = {
        'mean': float(np.mean(cog)),
        'std': float(np.std(cog)),
        'max': float(np.max(cog)),
        'min': float(np.min(cog)),
        'median': float(np.median(cog)),
        'skewness': float(stats.skew(cog)) if len(cog) > 2 else 0,
        'kurtosis': float(stats.kurtosis(cog)) if len(cog) > 2 else 0
    }
    
    # 总距离
    total_distance = float(np.sum(haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])))
    
    # 经纬度范围
    lat_range = float(np.max(lat) - np.min(lat))
    lon_range = float(np.max(lon) - np.min(lon))
    
    # 动态特征
    sog_diff = float(np.mean(np.abs(np.diff(sog)))) if point_count > 1 else 0
    cog_diff = float(np.mean(np.abs(np.diff(cog)))) if point_count > 1 else 0
    time_diffs = postime.diff().dt.total_seconds().values[1:]
    avg_time_interval = float(np.mean(time_diffs)) if len(time_diffs) > 0 else 0
    max_time_interval = float(np.max(time_diffs)) if len(time_diffs) > 0 else 0
    month_duration = float((postime.max() - postime.min()).total_seconds() / 3600) if point_count > 1 else 0
    avg_speed = total_distance / (avg_time_interval * point_count) if (avg_time_interval * point_count) > 0 else 0
    
    # 时间间隔熵
    def calculate_time_interval_entropy(time_diffs, n_bins=20):
        hist, _ = np.histogram(time_diffs, bins=n_bins, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0
    
    time_interval_entropy = calculate_time_interval_entropy(time_diffs)
    
    # 自相关系数
    def calculate_autocorrelation(data, lag=1):
        if len(data) < lag + 1:
            return 0
        return float(pd.Series(data).autocorr(lag=lag)) if not np.isnan(pd.Series(data).autocorr(lag=lag)) else 0
    
    sog_autocorrelation = calculate_autocorrelation(sog)
    cog_autocorrelation = calculate_autocorrelation(cog)
    
    # 分段特征（固定点数）
    segments = segment_trajectory(group, points_per_segment=100)
    segment_count = len(segments)
    segment_sog_vars = [seg['sog'].var() for seg in segments if len(seg) > 1]
    segment_cog_vars = [seg['cog'].var() for seg in segments if len(seg) > 1]
    segment_sog_var = float(np.mean(segment_sog_vars)) if segment_sog_vars else 0
    segment_cog_var = float(np.mean(segment_cog_vars)) if segment_cog_vars else 0
    
    # 距离分段特征（每100公里）
    def distance_segment_trajectory(group, distance_per_segment=100):
        segments = []
        group = group.sort_values('postime')
        current_distance = 0
        current_segment = []
        for i in range(1, len(group)):
            dist = haversine(group['lat'].iloc[i-1], group['lon'].iloc[i-1], group['lat'].iloc[i], group['lon'].iloc[i])
            current_distance += dist
            current_segment.append(i-1)
            if current_distance >= distance_per_segment:
                if len(current_segment) >= 2:
                    segments.append(group.iloc[current_segment])
                current_segment = []
                current_distance = 0
        if len(current_segment) >= 2:
            segments.append(group.iloc[current_segment])
        return segments
    
    distance_segments = distance_segment_trajectory(group, distance_per_segment=100)
    distance_segment_sog_vars = [seg['sog'].var() for seg in distance_segments if len(seg) > 1]
    distance_segment_cog_vars = [seg['cog'].var() for seg in distance_segments if len(seg) > 1]
    distance_segment_sog_var = float(np.mean(distance_segment_sog_vars)) if distance_segment_sog_vars else 0
    distance_segment_cog_var = float(np.mean(distance_segment_cog_vars)) if distance_segment_cog_vars else 0
    
    # 曲率计算
    def calculate_curvature(lat, lon, n=3):
        curvatures = []
        for i in range(n, len(lat) - n):
            p1 = (lat[i - n], lon[i - n])
            p2 = (lat[i], lon[i])
            p3 = (lat[i + n], lon[i + n])
            a = haversine(p1[0], p1[1], p2[0], p2[1])
            b = haversine(p2[0], p2[1], p3[0], p3[1])
            c = haversine(p1[0], p1[1], p3[0], p3[1])
            s = (a + b + c) / 2
            area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
            curvature = area / (a * b * c + 1e-10) if (a * b * c) > 0 else 0
            curvatures.append(float(curvature))
        return float(np.mean(curvatures)) if curvatures else 0, float(np.std(curvatures)) if curvatures else 0
    
    curvature_mean, curvature_std = calculate_curvature(lat, lon)
    
    # 转向频率
    def calculate_turn_frequency(cog, percentile=90):
        cog_changes = np.abs(np.diff(cog))
        threshold = np.percentile(cog_changes, percentile) if len(cog_changes) > 0 else 30
        turns = np.sum(cog_changes > threshold)
        return float(turns / len(cog)) if len(cog) > 0 else 0
    
    turn_frequency = calculate_turn_frequency(cog)
    
    # 急转弯次数
    def calculate_sharp_turn_count(cog, threshold=45):
        cog_changes = np.abs(np.diff(cog))
        return float(np.sum(cog_changes > threshold)) if len(cog_changes) > 0 else 0
    
    sharp_turn_count = calculate_sharp_turn_count(cog)
    
    # 停留点比例
    def calculate_dwell_point_ratio(sog, sog_threshold=1.0):
        return float(np.sum(sog < sog_threshold) / len(sog)) if len(sog) > 0 else 0
    
    dwell_point_ratio = calculate_dwell_point_ratio(sog)
    
    # 停靠区域数量（基于网格聚类）
    def calculate_dwell_regions(lat, lon, sog, sog_threshold=1.0, grid_size=0.05):
        dwell_points = sog < sog_threshold
        if np.sum(dwell_points) < 2:
            return 0, 0
        coords = np.vstack((lat[dwell_points], lon[dwell_points])).T
        grid_x = (coords[:, 0] // grid_size).astype(int)
        grid_y = (coords[:, 1] // grid_size).astype(int)
        unique_clusters = len(set(zip(grid_x, grid_y)))
        dwell_coords = coords
        dwell_point_spatial_mean = float(np.mean([haversine(coords[i, 0], coords[i, 1], np.mean(coords[:, 0]), np.mean(coords[:, 1])) for i in range(len(coords))])) if len(coords) > 0 else 0
        return unique_clusters, dwell_point_spatial_mean
    
    dwell_region_count, dwell_point_spatial_mean = calculate_dwell_regions(lat, lon, sog)
    
    # 轨迹形状复杂度（傅里叶描述符）
    def calculate_shape_complexity(lat, lon, n_freq=10):
        points = np.vstack((lon, lat)).T
        fft_coeffs = np.fft.fft(points[:, 0] + 1j * points[:, 1])
        low_freq = np.abs(fft_coeffs[1:n_freq//2]).mean() if len(fft_coeffs) > n_freq//2 else 0
        high_freq = np.abs(fft_coeffs[n_freq//2:n_freq]).mean() if len(fft_coeffs) > n_freq else 0
        return float(low_freq), float(high_freq)
    
    shape_complexity_lowfreq, shape_complexity_highfreq = calculate_shape_complexity(lat, lon)
    
    # 空间聚集度
    def calculate_clustering(lat, lon, grid_size=0.05):
        coords = np.vstack((lat, lon)).T
        grid_x = (coords[:, 0] // grid_size).astype(int)
        grid_y = (coords[:, 1] // grid_size).astype(int)
        labels = pd.Series(list(zip(grid_x, grid_y))).value_counts()
        max_cluster_ratio = labels.max() / len(coords) if len(labels) > 0 else 0
        n_clusters = len(labels)
        hist = labels / labels.sum() if len(labels) > 0 else np.array([])
        grid_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 and hist.sum() > 0 else 0
        return float(max_cluster_ratio), n_clusters, grid_entropy
    
    cluster_ratio, cluster_count, grid_entropy = calculate_clustering(lat, lon)
    
    # 方向一致性
    def calculate_directional_consistency(lat, lon):
        vectors = np.diff(np.vstack((lat, lon)).T, axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-10
        vectors = vectors[valid] / norms[valid][:, None]
        if len(vectors) < 2:
            return 0
        cos_sim = np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1, 1)
        return float(np.mean(cos_sim))
    
    directional_consistency = calculate_directional_consistency(lat, lon)
    
    # 凸包面积
    def calculate_convex_hull_area(lat, lon):
        try:
            from scipy.spatial import ConvexHull
            points = np.vstack((lon, lat)).T
            hull = ConvexHull(points)
            return float(hull.volume * (6371**2) * (np.pi/180)**2)
        except:
            return 0
    
    convex_hull_area = calculate_convex_hull_area(lat, lon)
    
    # 中心点偏移
    def calculate_centroid_shift(lat, lon):
        centroid = np.mean(np.vstack((lat, lon)).T, axis=0)
        start_point = np.array([lat[0], lon[0]])
        return float(haversine(centroid[0], centroid[1], start_point[0], start_point[1]))
    
    centroid_shift = calculate_centroid_shift(lat, lon)
    
    # 方向熵
    def calculate_directional_entropy(cog, n_bins=36):
        hist, _ = np.histogram(cog, bins=n_bins, range=(0, 360), density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0
    
    directional_entropy = calculate_directional_entropy(cog)
    
    # 航速模式特征
    def calculate_speed_features(sog):
        high_speed_ratio = float(np.sum(sog > 15) / len(sog)) if len(sog) > 0 else 0
        window_size = 5
        sog_smoothness = 0
        if len(sog) >= window_size:
            sog_windows = np.lib.stride_tricks.sliding_window_view(sog, window_size)
            sog_smoothness = float(np.mean(np.abs(np.diff(sog_windows, axis=1)).mean(axis=1)))
        hist, _ = np.histogram(sog, bins=20, range=(0, 50), density=True)
        hist = hist[hist > 0]
        sog_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0
        return high_speed_ratio, sog_smoothness, sog_entropy
    
    high_speed_ratio, sog_smoothness, sog_entropy = calculate_speed_features(sog)
    
    # 停靠行为特征
    def calculate_dwell_events(lat, lon, sog, postime, sog_threshold=1.0, min_duration=3600):
        dwell_points = sog < sog_threshold
        if np.sum(dwell_points) < 2:
            return 0, 0, 0
        dwell_times = postime[dwell_points]
        dwell_coords = np.vstack((lat[dwell_points], lon[dwell_points])).T
        time_diffs = dwell_times.diff().dt.total_seconds()
        events = []
        current_event = []
        for i, (t, diff) in enumerate(zip(dwell_times, time_diffs)):
            if pd.notnull(diff) and diff > min_duration:
                if current_event:
                    events.append(current_event)
                current_event = [t]
            elif pd.notnull(diff):
                current_event.append(t)
        if current_event:
            events.append(current_event)
        durations = [float((max(event) - min(event)).total_seconds() / 3600) for event in events if len(event) > 1]
        dwell_event_duration_mean = float(np.mean(durations)) if durations else 0
        total_time = (postime.max() - postime.min()).total_seconds() / 3600
        dwell_event_frequency = float(len(events) / total_time) if total_time > 0 else 0
        dwell_spatial_spread = 0
        if np.sum(dwell_points) >= 2:
            grid_size = 0.05
            grid_x = (dwell_coords[:, 0] // grid_size).astype(int)
            grid_y = (dwell_coords[:, 1] // grid_size).astype(int)
            cluster_sizes = pd.Series(list(zip(grid_x, grid_y))).value_counts()
            if len(cluster_sizes) > 1:
                probs = cluster_sizes / cluster_sizes.sum()
                dwell_spatial_spread = float(-np.sum(probs * np.log2(probs)))
        return dwell_event_duration_mean, dwell_event_frequency, dwell_spatial_spread
    
    dwell_event_duration_mean, dwell_event_frequency, dwell_spatial_spread = calculate_dwell_events(lat, lon, sog, postime)
    
    # 加速/减速事件
    def calculate_acceleration_events(sog, postime, threshold=5.0, min_duration=600):
        sog_diffs = np.abs(np.diff(sog))
        time_diffs = postime.diff().dt.total_seconds().values[1:]
        events = []
        current_event = []
        for i, (diff, t_diff) in enumerate(zip(sog_diffs, time_diffs)):
            if diff > threshold:
                current_event.append(i)
            elif len(current_event) > 0 and pd.notnull(t_diff) and t_diff > min_duration:
                if current_event:
                    events.append(current_event)
                current_event = []
        if current_event:
            events.append(current_event)
        durations = [float(postime.iloc[event[-1]].to_pydatetime().timestamp() - postime.iloc[event[0]].to_pydatetime().timestamp()) / 3600 for event in events if len(event) > 1]
        acceleration_event_count = float(len(events))
        acceleration_event_duration_mean = float(np.mean(durations)) if durations else 0
        return acceleration_event_count, acceleration_event_duration_mean
    
    acceleration_event_count, acceleration_event_duration_mean = calculate_acceleration_events(sog, postime)
    
    # 航线规律性特征
    def calculate_path_straightness(lat, lon):
        if len(lat) < 3:
            return 0
        start = np.array([lat[0], lon[0]])
        end = np.array([lat[-1], lon[-1]])
        straight_distance = haversine(start[0], start[1], end[0], end[1])
        return float(straight_distance / total_distance) if total_distance > 0 else 0
    
    path_straightness = calculate_path_straightness(lat, lon)
    
    # 网格重复访问频率
    def calculate_grid_repetition(lat, lon, grid_size=0.1):
        coords = np.vstack((lat, lon)).T
        grid_x = (coords[:, 1] // grid_size).astype(int)
        grid_y = (coords[:, 0] // grid_size).astype(int)
        grid_cells = list(zip(grid_x, grid_y))
        unique_cells = len(set(grid_cells))
        return float((len(grid_cells) - unique_cells) / len(grid_cells)) if len(grid_cells) > 0 else 0
    
    grid_repetition = calculate_grid_repetition(lat, lon)
    
    # 网格转换速度
    def calculate_grid_transition_speed(lat, lon, postime, grid_size=0.1):
        coords = np.vstack((lat, lon)).T
        grid_x = (coords[:, 1] // grid_size).astype(int)
        grid_y = (coords[:, 0] // grid_size).astype(int)
        grid_cells = list(zip(grid_x, grid_y))
        time_diffs = postime.diff().dt.total_seconds().values[1:]
        distances = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        transitions = [(dist / (t_diff / 3600)) for i, (dist, t_diff) in enumerate(zip(distances, time_diffs)) if i < len(grid_cells) - 1 and grid_cells[i] != grid_cells[i+1]]
        return float(np.mean(transitions)) if transitions else 0
    
    grid_transition_speed = calculate_grid_transition_speed(lat, lon, postime)
    
    # 主航向比例
    def calculate_dominant_cog_ratio(cog, n_bins=36):
        hist, _ = np.histogram(cog, bins=n_bins, range=(0, 360))
        return float(np.max(hist) / len(cog)) if len(cog) > 0 else 0
    
    dominant_cog_ratio = calculate_dominant_cog_ratio(cog)
    
    # 轨迹紧致度
    def calculate_trajectory_compactness(lat, lon):
        coords = np.vstack((lat, lon)).T
        centroid = np.mean(coords, axis=0)
        distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
        return float(np.mean(distances)) if len(distances) > 0 else 0
    
    trajectory_compactness = calculate_trajectory_compactness(lat, lon)
    
    # 空间自相关（Moran's I）
    def calculate_spatial_autocorrelation(lat, lon, sog):
        if len(sog) < 10:
            return 0
        coords = np.vstack((lat, lon)).T
        distances = np.sqrt(((coords[:, None] - coords) ** 2).sum(axis=2))
        weights = np.exp(-distances / 0.1)
        np.fill_diagonal(weights, 0)
        z = sog - np.mean(sog)
        numerator = np.sum(weights * np.outer(z, z))
        denominator = np.sum(weights) * np.var(sog)
        return float(numerator / denominator) if denominator > 0 else 0
    
    spatial_autocorrelation = calculate_spatial_autocorrelation(lat, lon, sog)
    
    # 分形维度
    def calculate_fractal_dimension(lat, lon):
        if len(lat) < 10:
            return 0
        coords = np.vstack((lat, lon)).T
        scales = np.logspace(0, 2, num=5)
        counts = []
        for scale in scales:
            grid_x = (coords[:, 1] // scale).astype(int)
            grid_y = (coords[:, 0] // scale).astype(int)
            grid_cells = set(zip(grid_x, grid_y))
            counts.append(len(grid_cells))
        counts = np.array(counts)
        scales = np.array(scales)
        coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
        return float(coeffs[0])
    
    fractal_dimension = calculate_fractal_dimension(lat, lon)
    
    # 傅里叶变换特征（SOG 和 COG）
    def calculate_fft_features(data, n_freq=15):
        fft_coeffs = np.abs(np.fft.fft(data - np.mean(data)))
        low_freq = float(np.mean(fft_coeffs[1:n_freq//3])) if len(fft_coeffs) > n_freq//3 else 0
        mid_freq = float(np.mean(fft_coeffs[n_freq//3:2*n_freq//3])) if len(fft_coeffs) > 2*n_freq//3 else 0
        high_freq = float(np.mean(fft_coeffs[2*n_freq//3:n_freq])) if len(fft_coeffs) > n_freq else 0
        return low_freq, mid_freq, high_freq
    
    fft_sog_power_low, fft_sog_power_mid, fft_sog_power_high = calculate_fft_features(sog)
    fft_cog_power_low, fft_cog_power_mid, fft_cog_power_high = calculate_fft_features(cog)
    
    # 速度与航向相关性
    def calculate_sog_cog_correlation(sog, cog):
        if len(sog) < 2:
            return 0
        return float(np.corrcoef(sog, cog)[0, 1]) if not np.isnan(np.corrcoef(sog, cog)[0, 1]) else 0
    
    sog_cog_correlation = calculate_sog_cog_correlation(sog, cog)
    
    # 加速度特征
    def calculate_acceleration_features(sog, postime):
        time_diffs = postime.diff().dt.total_seconds().values[1:]
        sog_diff = np.diff(sog)
        acceleration = sog_diff / (time_diffs + 1e-10)
        return float(np.mean(np.abs(acceleration))) if len(acceleration) > 0 else 0, float(np.std(acceleration)) if len(acceleration) > 0 else 0
    
    sog_acceleration_mean, sog_acceleration_std = calculate_acceleration_features(sog, postime)
    
    # 航向变化熵
    def calculate_cog_change_entropy(cog, n_bins=36):
        cog_changes = np.abs(np.diff(cog))
        hist, _ = np.histogram(cog_changes, bins=n_bins, range=(0, 180), density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0
    
    cog_change_entropy = calculate_cog_change_entropy(cog)
    
    # 网格转换频率
    def calculate_grid_transition_freq(lat, lon, grid_size=0.1):
        coords = np.vstack((lat, lon)).T
        grid_x = (coords[:, 1] // grid_size).astype(int)
        grid_y = (coords[:, 0] // grid_size).astype(int)
        grid_cells = list(zip(grid_x, grid_y))
        transitions = sum(1 for i in range(1, len(grid_cells)) if grid_cells[i] != grid_cells[i-1])
        return float(transitions / (len(grid_cells) - 1)) if len(grid_cells) > 1 else 0
    
    grid_transition_freq = calculate_grid_transition_freq(lat, lon)
    
    # 轨迹自交叉点数量
    def calculate_trajectory_crossings(lat, lon):
        crossings = 0
        for i in range(len(lat) - 3):
            for j in range(i + 2, len(lat) - 1):
                p1, p2 = (lat[i], lon[i]), (lat[i+1], lon[i+1])
                p3, p4 = (lat[j], lon[j]), (lat[j+1], lon[j+1])
                def ccw(A, B, C):
                    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
                    crossings += 1
        return float(crossings)
    
    trajectory_crossings = calculate_trajectory_crossings(lat, lon)
    
    # K-Means 聚类密度
    def calculate_kmeans_cluster_density(lat, lon, n_clusters=5):
        if len(lat) < n_clusters:
            return 0
        coords = np.vstack((lat, lon)).T
        kmeans = KMeans(n_clusters=min(n_clusters, len(coords)), random_state=42)
        kmeans.fit(coords)
        labels = kmeans.labels_
        label_counts = pd.Series(labels).value_counts()
        max_density = label_counts.max() / len(coords) if len(label_counts) > 0 else 0
        return float(max_density)
    
    kmeans_cluster_density = calculate_kmeans_cluster_density(lat, lon)
    
    # 直线段平均长度
    def calculate_straight_segment_length(segments):
        straight_lengths = []
        for segment in segments:
            if len(segment) < 3:
                continue
            seg_lat = segment['lat'].values
            seg_lon = segment['lon'].values
            start = np.array([seg_lat[0], seg_lon[0]])
            end = np.array([seg_lat[-1], seg_lon[-1]])
            straight_distance = haversine(start[0], start[1], end[0], end[1])
            total_seg_distance = np.sum(haversine(seg_lat[:-1], seg_lon[:-1], seg_lat[1:], seg_lon[1:]))
            if total_seg_distance > 0 and straight_distance / total_seg_distance > 0.9:
                straight_lengths.append(total_seg_distance)
        return float(np.mean(straight_lengths)) if straight_lengths else 0
    
    straight_segment_length_mean = calculate_straight_segment_length(segments)
    
    # 航行模式比例（直线和环形）
    def calculate_segment_patterns(lat, lon):
        straight_segments = 0
        loop_segments = 0
        for segment in segments:
            if len(segment) < 3:
                continue
            seg_lat = segment['lat'].values
            seg_lon = segment['lon'].values
            start = np.array([seg_lat[0], seg_lon[0]])
            end = np.array([seg_lat[-1], seg_lon[-1]])
            straight_distance = haversine(start[0], start[1], end[0], end[1])
            total_seg_distance = np.sum(haversine(seg_lat[:-1], seg_lon[:-1], seg_lat[1:], seg_lon[1:]))
            if total_seg_distance > 0 and straight_distance / total_seg_distance > 0.9:
                straight_segments += 1
            elif haversine(seg_lat[0], seg_lon[0], seg_lat[-1], seg_lon[-1]) < 10:  # 环形判断
                loop_segments += 1
        total_segments = len(segments)
        straight_ratio = float(straight_segments / total_segments) if total_segments > 0 else 0
        loop_ratio = float(loop_segments / total_segments) if total_segments > 0 else 0
        return straight_ratio, loop_ratio
    
    straight_segment_ratio, loop_segment_ratio = calculate_segment_patterns(lat, lon)
    
    # 更新特征字典
    result.update({
        'sog_mean': sog_stats['mean'], 'sog_std': sog_stats['std'],
        'sog_max': sog_stats['max'], 'sog_min': sog_stats['min'], 'sog_median': sog_stats['median'],
        'cog_mean': cog_stats['mean'], 'cog_std': cog_stats['std'],
        'cog_max': cog_stats['max'], 'cog_min': cog_stats['min'], 'cog_median': cog_stats['median'],
        'point_count': point_count,
        'total_distance': total_distance,
        'avg_speed': avg_speed,
        'sog_change_rate': sog_diff,
        'cog_change_rate': cog_diff,
        'avg_time_interval': avg_time_interval,
        'segment_count': segment_count,
        'segment_sog_var': segment_sog_var,
        'segment_cog_var': segment_cog_var,
        'distance_segment_sog_var': distance_segment_sog_var,
        'distance_segment_cog_var': distance_segment_cog_var,
        'lat_range': lat_range,
        'lon_range': lon_range,
        'curvature_mean': curvature_mean,
        'curvature_std': curvature_std,
        'turn_frequency': turn_frequency,
        'dwell_point_ratio': dwell_point_ratio,
        'dwell_region_count': dwell_region_count,
        'month_duration': month_duration,
        'shape_complexity_lowfreq': shape_complexity_lowfreq,
        'shape_complexity_highfreq': shape_complexity_highfreq,
        'cluster_ratio': cluster_ratio,
        'cluster_count': cluster_count,
        'directional_consistency': directional_consistency,
        'convex_hull_area': convex_hull_area,
        'directional_entropy': directional_entropy,
        'high_speed_ratio': high_speed_ratio,
        'sog_smoothness': sog_smoothness,
        'sog_entropy': sog_entropy,
        'dwell_event_duration_mean': dwell_event_duration_mean,
        'dwell_event_frequency': dwell_event_frequency,
        'dwell_spatial_spread': dwell_spatial_spread,
        'path_straightness': path_straightness,
        'grid_repetition': grid_repetition,
        'dominant_cog_ratio': dominant_cog_ratio,
        'trajectory_compactness': trajectory_compactness,
        'spatial_autocorrelation': spatial_autocorrelation,
        'fractal_dimension': fractal_dimension,
        'fft_sog_power_low': fft_sog_power_low,
        'fft_sog_power_high': fft_sog_power_high,
        'fft_cog_power_low': fft_cog_power_low,
        'fft_cog_power_high': fft_cog_power_high,
        'sog_acceleration_mean': sog_acceleration_mean,
        'sog_acceleration_std': sog_acceleration_std,
        'cog_change_entropy': cog_change_entropy,
        'grid_transition_freq': grid_transition_freq,
        'straight_segment_ratio': straight_segment_ratio,
        'loop_segment_ratio': loop_segment_ratio,
        'sog_skewness': sog_stats['skewness'],
        'sog_kurtosis': sog_stats['kurtosis'],
        'cog_skewness': cog_stats['skewness'],
        'cog_kurtosis': cog_stats['kurtosis'],
        'time_interval_entropy': time_interval_entropy,
        'max_time_interval': max_time_interval,
        'sog_autocorrelation': sog_autocorrelation,
        'cog_autocorrelation': cog_autocorrelation,
        'trajectory_crossings': trajectory_crossings,
        'grid_density': cluster_ratio,
        'grid_entropy': grid_entropy,
        'centroid_shift': centroid_shift,
        'acceleration_event_count': acceleration_event_count,
        'acceleration_event_duration_mean': acceleration_event_duration_mean,
        'sharp_turn_count': sharp_turn_count,
        'straight_segment_length_mean': straight_segment_length_mean,
        'dwell_point_spatial_mean': dwell_point_spatial_mean,
        'fft_sog_power_mid': fft_sog_power_mid,
        'fft_cog_power_mid': fft_cog_power_mid,
        'sog_cog_correlation': sog_cog_correlation,
        'grid_transition_speed': grid_transition_speed,
        'kmeans_cluster_density': kmeans_cluster_density
    })
    
    return result

def extract_features_parallel(df, is_train=True, n_jobs=None):
    if is_train:
        grouped = list(df.groupby(['shipno', 'month_split', 'split_type']))
    else:
        grouped = list(df.groupby('shipno'))

    n_jobs = n_jobs or Config().n_jobs
    print(f"🚀 开始并行提取 {len(grouped)} 条船舶轨迹特征（使用 {n_jobs} 核，is_train={is_train}）...")
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap(_extract_single_ship_feature, grouped), total=len(grouped)))
    features_df = pd.DataFrame(results)
    logging.info(f"提取的特征类型:\n{features_df.dtypes}")
    logging.info(f"提取的表格特征样本数: {len(features_df)}")
    return features_df
