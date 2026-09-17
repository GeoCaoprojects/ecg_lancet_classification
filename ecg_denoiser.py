from __future__ import annotations

from typing import Sequence

import numpy as np
import pywt
import wfdb
import sys
import os

class ECGWaveletKalmanBilateralDenoiser:
    """
    Morphology-preserving ECG denoising pipeline.

    Pipeline:
        raw ECG
          -> long reflected context padding
          -> two-stage median baseline correction
          -> stationary wavelet transform (SWT) using PyWavelets
          -> frequency-aware adaptive Kalman/RTS smoothing on detail bands
             with level-dependent QRS and large-coefficient protection
          -> inverse SWT
          -> mild morphology-aware bilateral shrinkage outside QRS regions
          -> robust isoelectric-line centering
          -> denoised ECG

    The public denoise() method returns only the final ECG samples as list[float].
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        wavelet: str = "sym4",
        level: int = 4,
        baseline_short_window_s: float = 0.20,
        baseline_long_window_s: float = 0.60,
        edge_padding_s: float = 1.50,
        kalman_measurement_scale: float = 1.0,
        coefficient_protect_low_sigma: float = 2.5,
        coefficient_protect_high_sigma: float = 5.0,
        coefficient_protection_strength: float = 0.30,
        qrs_gradient_sigma: float = 4.0,
        qrs_protection_radius_s: float = 0.055,
        qrs_protection_by_level: Sequence[float] = (0.40, 0.55, 0.75, 0.90),
        bilateral_radius_s: float = 0.030,
        bilateral_sigma_spatial_s: float = 0.015,
        bilateral_sigma_range: float | None = None,
        bilateral_range_noise_scale: float = 2.5,
        shrinkage_threshold_scale: float = 0.30,
        shrinkage_strength: float = 0.12,
        center_output: bool = True,
    ) -> None:
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be > 0")
        if level < 1:
            raise ValueError("level must be >= 1")
        if baseline_short_window_s <= 0:
            raise ValueError("baseline_short_window_s must be > 0")
        if baseline_long_window_s <= baseline_short_window_s:
            raise ValueError(
                "baseline_long_window_s must exceed baseline_short_window_s"
            )
        if edge_padding_s < 0:
            raise ValueError("edge_padding_s must be >= 0")
        if kalman_measurement_scale <= 0:
            raise ValueError("kalman_measurement_scale must be > 0")
        if coefficient_protect_low_sigma < 0:
            raise ValueError("coefficient_protect_low_sigma must be >= 0")
        if coefficient_protect_high_sigma <= coefficient_protect_low_sigma:
            raise ValueError(
                "coefficient_protect_high_sigma must exceed coefficient_protect_low_sigma"
            )
        if not 0.0 <= coefficient_protection_strength <= 1.0:
            raise ValueError("coefficient_protection_strength must be in [0, 1]")
        if qrs_gradient_sigma <= 0:
            raise ValueError("qrs_gradient_sigma must be > 0")
        if qrs_protection_radius_s < 0:
            raise ValueError("qrs_protection_radius_s must be >= 0")
        if len(qrs_protection_by_level) < 1:
            raise ValueError("qrs_protection_by_level must not be empty")
        if any(not 0.0 <= float(value) <= 1.0 for value in qrs_protection_by_level):
            raise ValueError("all qrs_protection_by_level values must be in [0, 1]")
        if bilateral_radius_s <= 0 or bilateral_sigma_spatial_s <= 0:
            raise ValueError("bilateral time parameters must be > 0")
        if bilateral_sigma_range is not None and bilateral_sigma_range <= 0:
            raise ValueError("bilateral_sigma_range must be > 0 when specified")
        if bilateral_range_noise_scale <= 0:
            raise ValueError("bilateral_range_noise_scale must be > 0")
        if shrinkage_threshold_scale < 0:
            raise ValueError("shrinkage_threshold_scale must be >= 0")
        if not 0.0 <= shrinkage_strength <= 1.0:
            raise ValueError("shrinkage_strength must be in [0, 1]")

        pywt.Wavelet(wavelet)

        self.sampling_rate_hz = float(sampling_rate_hz)
        self.wavelet = wavelet
        self.level = int(level)
        self.baseline_short_window_s = float(baseline_short_window_s)
        self.baseline_long_window_s = float(baseline_long_window_s)
        self.edge_padding_s = float(edge_padding_s)
        self.kalman_measurement_scale = float(kalman_measurement_scale)
        self.coefficient_protect_low_sigma = float(coefficient_protect_low_sigma)
        self.coefficient_protect_high_sigma = float(coefficient_protect_high_sigma)
        self.coefficient_protection_strength = float(
            coefficient_protection_strength
        )
        self.qrs_gradient_sigma = float(qrs_gradient_sigma)
        self.qrs_protection_radius_s = float(qrs_protection_radius_s)
        self.qrs_protection_by_level = tuple(
            float(value) for value in qrs_protection_by_level
        )
        self.bilateral_radius_s = float(bilateral_radius_s)
        self.bilateral_sigma_spatial_s = float(bilateral_sigma_spatial_s)
        self.bilateral_sigma_range = bilateral_sigma_range
        self.bilateral_range_noise_scale = float(bilateral_range_noise_scale)
        self.shrinkage_threshold_scale = float(shrinkage_threshold_scale)
        self.shrinkage_strength = float(shrinkage_strength)
        self.center_output = bool(center_output)

    @staticmethod
    def _as_signal(ecg: Sequence[float]) -> np.ndarray:
        x = np.asarray(ecg, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("ECG input must be one-dimensional")
        if x.size < 32:
            raise ValueError("ECG input is too short")
        if not np.all(np.isfinite(x)):
            raise ValueError("ECG input contains NaN or infinite values")
        return x

    @staticmethod
    def _robust_sigma(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return float(np.finfo(np.float64).eps)
        center = np.median(values)
        mad = np.median(np.abs(values - center))
        sigma = mad / 0.6744897501960817
        return float(max(sigma, np.finfo(np.float64).eps))

    def _window_samples(self, duration_s: float, signal_size: int) -> int:
        """Convert a duration to an odd window length bounded by the signal size."""
        window = max(3, int(round(duration_s * self.sampling_rate_hz)))
        if window % 2 == 0:
            window += 1

        largest_odd = signal_size if signal_size % 2 == 1 else signal_size - 1
        window = min(window, largest_odd)
        return max(3, window)

    @staticmethod
    def _median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
        """One-dimensional reflected median filter implemented with NumPy."""
        if window <= 1:
            return x.copy()
        if window % 2 == 0:
            raise ValueError("median filter window must be odd")

        radius = window // 2
        padded = np.pad(x, (radius, radius), mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        return np.median(windows, axis=-1)

    def _remove_baseline(self, x: np.ndarray) -> np.ndarray:
        """
        Estimate baseline wander with two median filters and subtract it.

        The short window suppresses QRS complexes in the baseline estimate.
        The long window suppresses P and T waves while retaining slow drift.
        """
        short_window = self._window_samples(
            self.baseline_short_window_s,
            x.size,
        )
        long_window = self._window_samples(
            self.baseline_long_window_s,
            x.size,
        )

        first_stage = self._median_filter_1d(x, short_window)
        baseline = self._median_filter_1d(first_stage, long_window)
        return x - baseline

    def _pad_context(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Add long reflected context before baseline and wavelet processing.

        This substantially reduces beginning and ending artifacts after baseline
        correction, SWT reconstruction, and bilateral processing.
        """
        if self.edge_padding_s <= 0:
            return x.copy(), 0

        requested = max(
            1,
            int(round(self.edge_padding_s * self.sampling_rate_hz)),
        )
        padding = min(requested, x.size - 1)
        padded = np.pad(x, (padding, padding), mode="reflect")
        return padded, padding

    def _pad_for_swt(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Pad symmetrically so the signal length is divisible by 2**level,
        which is required by the stationary wavelet transform.
        """
        block = 2 ** self.level
        remainder = x.size % block
        if remainder == 0:
            return x.copy(), 0

        total_padding = block - remainder
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        padded = np.pad(
            x,
            (left_padding, right_padding),
            mode="reflect",
        )
        return padded, left_padding

    @staticmethod
    def _max_filter_1d(x: np.ndarray, radius: int) -> np.ndarray:
        """Local maximum filter used to widen QRS protection regions."""
        if radius <= 0:
            return x.copy()
        window = 2 * radius + 1
        padded = np.pad(x, (radius, radius), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        return np.max(windows, axis=-1)

    @staticmethod
    def _moving_average_1d(x: np.ndarray, radius: int) -> np.ndarray:
        """Short moving average used only for smooth mask transitions."""
        if radius <= 0:
            return x.copy()
        window = 2 * radius + 1
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(x, kernel, mode="same")

    def _create_qrs_protection_mask(self, x: np.ndarray) -> np.ndarray:
        """
        Detect fast morphology changes and create a soft protection mask.

        The mask is near 1 around QRS complexes and near 0 in flatter regions.
        It is intentionally conservative and is not a clinical QRS detector.
        """
        detector = np.convolve(
            x,
            np.ones(3, dtype=np.float64) / 3.0,
            mode="same",
        )
        gradient = np.abs(np.gradient(detector))
        gradient_center = float(np.median(gradient))
        gradient_sigma = self._robust_sigma(gradient)
        threshold = gradient_center + self.qrs_gradient_sigma * gradient_sigma

        seed = (gradient > threshold).astype(np.float64)
        radius = max(
            1,
            int(round(self.qrs_protection_radius_s * self.sampling_rate_hz)),
        )
        expanded = self._max_filter_1d(seed, radius)
        softened = self._moving_average_1d(expanded, max(1, radius // 2))
        return np.clip(softened, 0.0, 1.0)

    def _swt(
        self,
        x: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Stationary Wavelet Transform using PyWavelets.

        The returned list is ordered from the coarsest requested level to D1:
            [(cA_L, cD_L), ..., (cA_1, cD_1)]
        """
        coeffs = pywt.swt(
            x,
            wavelet=self.wavelet,
            level=self.level,
            trim_approx=False,
            norm=False,
        )
        return [
            (
                np.asarray(approximation, dtype=np.float64),
                np.asarray(detail, dtype=np.float64),
            )
            for approximation, detail in coeffs
        ]

    def _kalman_process_ratio(self, detail_level: int) -> float:
        """
        Select Kalman strength from the approximate physical frequency band.

        Smaller process ratios give stronger smoothing. D1 and D2 receive more
        smoothing than the morphology-rich lower-frequency detail bands.
        """
        low_hz = self.sampling_rate_hz / (2 ** (detail_level + 1))
        high_hz = self.sampling_rate_hz / (2 ** detail_level)

        if low_hz >= 40.0:
            return 0.04
        if low_hz >= 20.0:
            return 0.07
        if low_hz >= 10.0:
            return 0.20
        if high_hz >= 8.0:
            return 0.65
        return 1.50

    def _qrs_protection_weight(self, detail_level: int) -> float:
        """Return the level-dependent QRS protection weight for a detail band."""
        index = min(
            max(detail_level - 1, 0),
            len(self.qrs_protection_by_level) - 1,
        )
        return self.qrs_protection_by_level[index]

    def _kalman_rts_smoother(
        self,
        measurements: np.ndarray,
        noise_sigma: float,
        process_ratio: float,
    ) -> np.ndarray:
        """
        Scalar random-walk Kalman filter followed by RTS fixed-interval smoothing.
        """
        z = np.asarray(measurements, dtype=np.float64)
        if z.size <= 2:
            return z.copy()

        measurement_variance = max(
            (self.kalman_measurement_scale * noise_sigma) ** 2,
            np.finfo(np.float64).eps,
        )
        process_variance = max(
            process_ratio * measurement_variance,
            np.finfo(np.float64).eps,
        )

        filtered_state = np.empty_like(z)
        filtered_covariance = np.empty_like(z)
        predicted_state = np.empty_like(z)
        predicted_covariance = np.empty_like(z)

        filtered_state[0] = z[0]
        filtered_covariance[0] = measurement_variance
        predicted_state[0] = z[0]
        predicted_covariance[0] = measurement_variance

        for i in range(1, z.size):
            predicted_state[i] = filtered_state[i - 1]
            predicted_covariance[i] = (
                filtered_covariance[i - 1] + process_variance
            )

            gain = predicted_covariance[i] / (
                predicted_covariance[i] + measurement_variance
            )
            filtered_state[i] = predicted_state[i] + gain * (
                z[i] - predicted_state[i]
            )
            filtered_covariance[i] = (1.0 - gain) * predicted_covariance[i]

        smoothed_state = filtered_state.copy()

        for i in range(z.size - 2, -1, -1):
            denominator = max(
                predicted_covariance[i + 1],
                np.finfo(np.float64).eps,
            )
            smoother_gain = filtered_covariance[i] / denominator
            smoothed_state[i] = filtered_state[i] + smoother_gain * (
                smoothed_state[i + 1] - predicted_state[i + 1]
            )

        return smoothed_state

    def _adaptive_kalman_detail(
        self,
        detail: np.ndarray,
        detail_level: int,
        qrs_mask: np.ndarray,
        noise_sigma: float,
    ) -> np.ndarray:
        """
        Smooth a detail band with level-dependent morphology protection.

        D1 and D2 retain enough QRS information to preserve timing and shape,
        but do not preserve every high-frequency fluctuation around the R peak.
        D3 and higher levels receive stronger morphology protection.
        """
        process_ratio = self._kalman_process_ratio(detail_level)
        kalman = self._kalman_rts_smoother(
            detail,
            noise_sigma=noise_sigma,
            process_ratio=process_ratio,
        )

        local_sigma = self._robust_sigma(detail)
        low = self.coefficient_protect_low_sigma * local_sigma
        high = self.coefficient_protect_high_sigma * local_sigma
        coefficient_protection = (np.abs(detail) - low) / max(
            high - low,
            np.finfo(np.float64).eps,
        )
        coefficient_protection = np.clip(coefficient_protection, 0.0, 1.0)

        qrs_weight = self._qrs_protection_weight(detail_level)
        protection = np.maximum(
            self.coefficient_protection_strength * coefficient_protection,
            qrs_weight * qrs_mask,
        )

        return protection * detail + (1.0 - protection) * kalman

    def _inverse_swt(
        self,
        coefficients: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        reconstructed = pywt.iswt(
            coefficients,
            wavelet=self.wavelet,
            norm=False,
        )
        return np.asarray(reconstructed, dtype=np.float64)

    def _bilateral_filter_1d(
        self,
        x: np.ndarray,
        sigma_range: float,
    ) -> np.ndarray:
        radius = max(
            1,
            int(round(self.bilateral_radius_s * self.sampling_rate_hz)),
        )
        sigma_space = max(
            1.0,
            self.bilateral_sigma_spatial_s * self.sampling_rate_hz,
        )

        out = np.empty_like(x)
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        spatial_template = np.exp(-0.5 * (offsets / sigma_space) ** 2)

        for i in range(x.size):
            left = max(0, i - radius)
            right = min(x.size, i + radius + 1)
            neighborhood = x[left:right]

            template_left = left - (i - radius)
            template_right = template_left + neighborhood.size
            spatial_weights = spatial_template[template_left:template_right]

            amplitude_delta = neighborhood - x[i]
            range_weights = np.exp(-0.5 * (amplitude_delta / sigma_range) ** 2)
            weights = spatial_weights * range_weights
            weight_sum = float(np.sum(weights))

            if weight_sum <= np.finfo(np.float64).eps:
                out[i] = x[i]
            else:
                out[i] = float(np.dot(weights, neighborhood) / weight_sum)

        return out

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

    def _morphology_aware_bilateral_shrinkage(
        self,
        reconstructed: np.ndarray,
        noise_sigma: float,
        qrs_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply mild bilateral shrinkage outside QRS regions.

        Wavelet-domain processing performs most of the denoising. This final
        stage only reduces small residual fluctuations in non-QRS regions.
        """
        if self.bilateral_sigma_range is None:
            sigma_range = max(
                self.bilateral_range_noise_scale * noise_sigma,
                np.finfo(np.float64).eps,
            )
        else:
            sigma_range = float(self.bilateral_sigma_range)

        bilateral = self._bilateral_filter_1d(
            reconstructed,
            sigma_range=sigma_range,
        )

        residual = reconstructed - bilateral
        threshold = self.shrinkage_threshold_scale * noise_sigma
        shrunk_residual = self._soft_threshold(residual, threshold)
        shrinkage_candidate = bilateral + shrunk_residual

        effective_strength = self.shrinkage_strength * (1.0 - qrs_mask)

        return (
            (1.0 - effective_strength) * reconstructed
            + effective_strength * shrinkage_candidate
        )

    def _center_stable_baseline(
        self,
        x: np.ndarray,
        qrs_mask: np.ndarray,
    ) -> np.ndarray:
        """Remove only a residual global DC offset using stable non-QRS samples."""
        if not self.center_output:
            return x.copy()

        non_qrs = qrs_mask < 0.20
        if np.count_nonzero(non_qrs) < 16:
            return x - np.median(x)

        gradient = np.abs(np.gradient(x))
        gradient_limit = float(np.quantile(gradient[non_qrs], 0.35))
        stable = non_qrs & (gradient <= gradient_limit)

        if np.count_nonzero(stable) < 16:
            stable = non_qrs

        first_center = float(np.median(x[stable]))
        stable_sigma = self._robust_sigma(x[stable])
        inlier = stable & (
            np.abs(x - first_center) <= 3.0 * stable_sigma
        )

        if np.count_nonzero(inlier) >= 16:
            center = float(np.median(x[inlier]))
        else:
            center = first_center

        return x - center

    def denoise(self, ecg: Sequence[float]) -> list[float]:
        """Return only the final denoised ECG samples as a Python list."""
        raw = self._as_signal(ecg)

        # 1. Add long reflected context to suppress beginning and ending artifacts.
        context_padded, context_padding = self._pad_context(raw)

        # 2. Remove baseline wander using the two-stage median estimator.
        corrected_context = self._remove_baseline(context_padded)

        # 3. Pad only as needed for SWT length divisibility.
        swt_padded, swt_left_padding = self._pad_for_swt(corrected_context)
        qrs_mask = self._create_qrs_protection_mask(swt_padded)

        # 4. Stationary wavelet decomposition.
        coefficients = self._swt(swt_padded)

        # The last tuple is level 1, so its detail array is cD1.
        noise_sigma = self._robust_sigma(coefficients[-1][1])

        # 5. Frequency-aware adaptive Kalman/RTS smoothing on detail bands.
        filtered_coefficients: list[tuple[np.ndarray, np.ndarray]] = []
        for index, (approximation, detail) in enumerate(coefficients):
            detail_level = self.level - index
            filtered_detail = self._adaptive_kalman_detail(
                detail,
                detail_level=detail_level,
                qrs_mask=qrs_mask,
                noise_sigma=noise_sigma,
            )
            filtered_coefficients.append((approximation, filtered_detail))

        # 6. Inverse SWT.
        reconstructed = self._inverse_swt(filtered_coefficients)

        # 7. Mild morphology-aware bilateral shrinkage outside QRS regions.
        denoised_swt = self._morphology_aware_bilateral_shrinkage(
            reconstructed,
            noise_sigma=noise_sigma,
            qrs_mask=qrs_mask,
        )

        # 8. Remove the small SWT divisibility padding.
        context_size = corrected_context.size
        denoised_context = denoised_swt[
            swt_left_padding : swt_left_padding + context_size
        ]
        qrs_context = qrs_mask[
            swt_left_padding : swt_left_padding + context_size
        ]

        # 9. Remove the long reflected context padding.
        denoised = denoised_context[
            context_padding : context_padding + raw.size
        ]
        qrs_original = qrs_context[
            context_padding : context_padding + raw.size
        ]

        # 10. Remove only a residual global baseline offset.
        denoised = self._center_stable_baseline(
            denoised,
            qrs_original,
        )

        return denoised.tolist()

