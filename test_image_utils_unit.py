from __future__ import annotations

from io import BytesIO
import os
import unittest
from unittest.mock import patch

from PIL import Image


_REQUIRED_ENV = {
    "DB_USER": "test",
    "DB_PASSWORD": "test",
    "DB_HOST": "127.0.0.1",
    "DB_PORT": "5432",
    "DB_NAME": "test",
    "DB_SCHEME": "postgresql",
    "DEBUG": "false",
    "ARCFACE_PROVIDERS": "CPUExecutionProvider",
    "ARCFACE_MODEL_NAME": "buffalo_l",
    "ARCFACE_DET_SIZE": "640",
    "MATCH_MIN_SIMILARITY": "0.45",
    "JWT_ACCESS_SECRET": "test-jwt-secret",
    "AI_INTERNAL_TOKEN": "test-internal-token",
    "PHOTO_API_URL": "http://127.0.0.1:8080/api/file/photo",
    "PHOTO_API_TIMEOUT_SECONDS": "1",
}
for _key, _value in _REQUIRED_ENV.items():
    os.environ.setdefault(_key, _value)

from app.utils.image_utils import decode_image_bytes, inspect_image_dimensions


def _one_pixel_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(output, format="PNG")
    return output.getvalue()


def _oriented_jpeg(orientation: int) -> bytes:
    output = BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    image = Image.new("RGB", (40, 20))
    image.paste((255, 0, 0), (0, 0, 20, 20))
    image.paste((0, 0, 255), (20, 0, 40, 20))
    image.save(
        output,
        format="JPEG",
        exif=exif,
        quality=100,
        subsampling=0,
    )
    return output.getvalue()


class ImageDecodeLimitTest(unittest.TestCase):
    def assert_bgr_close(self, actual, expected, delta=10):
        self.assertTrue(
            all(abs(int(component) - target) <= delta for component, target in zip(actual, expected)),
            f"expected BGR {expected}, got {actual.tolist()}",
        )

    def test_valid_png_is_inspected_and_decoded(self):
        encoded = _one_pixel_png()

        self.assertEqual({"width": 1, "height": 1}, inspect_image_dimensions(encoded, max_pixels=1))
        decoded = decode_image_bytes(encoded, max_pixels=1)

        self.assertIsNotNone(decoded)
        self.assertEqual((1, 1, 3), decoded.shape)

    def test_oversized_header_is_rejected_before_opencv_decode(self):
        encoded = bytearray(_one_pixel_png())
        encoded[16:20] = (10_000).to_bytes(4, "big")
        encoded[20:24] = (10_000).to_bytes(4, "big")

        with patch("app.utils.image_utils.cv2.imdecode") as imdecode:
            self.assertIsNone(decode_image_bytes(bytes(encoded), max_pixels=60_000_000))
            imdecode.assert_not_called()

    def test_truncated_image_is_rejected(self):
        self.assertIsNone(inspect_image_dimensions(_one_pixel_png()[:24]))

    def test_exif_orientation_6_is_rotated_clockwise(self):
        encoded = _oriented_jpeg(6)

        self.assertEqual({"width": 40, "height": 20}, inspect_image_dimensions(encoded))
        decoded = decode_image_bytes(encoded)

        self.assertIsNotNone(decoded)
        self.assertEqual((40, 20, 3), decoded.shape)
        self.assert_bgr_close(decoded[5, 10], (0, 0, 255))
        self.assert_bgr_close(decoded[35, 10], (255, 0, 0))

    def test_exif_orientation_8_is_rotated_counterclockwise(self):
        encoded = _oriented_jpeg(8)

        self.assertEqual({"width": 40, "height": 20}, inspect_image_dimensions(encoded))
        decoded = decode_image_bytes(encoded)

        self.assertIsNotNone(decoded)
        self.assertEqual((40, 20, 3), decoded.shape)
        self.assert_bgr_close(decoded[5, 10], (255, 0, 0))
        self.assert_bgr_close(decoded[35, 10], (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
