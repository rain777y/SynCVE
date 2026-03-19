"""Create static test images using PIL (no API needed)."""
from PIL import Image, ImageDraw
from pathlib import Path


def create_test_images():
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # 1. Basic face-like image
    img = Image.new("RGB", (640, 480), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)
    # Face oval
    draw.ellipse(
        [220, 80, 420, 400], fill=(220, 195, 175), outline=(180, 160, 140), width=2
    )
    # Eyes
    draw.ellipse([270, 180, 310, 210], fill=(255, 255, 255))
    draw.ellipse([280, 188, 300, 202], fill=(60, 40, 20))
    draw.ellipse([330, 180, 370, 210], fill=(255, 255, 255))
    draw.ellipse([340, 188, 360, 202], fill=(60, 40, 20))
    # Nose
    draw.polygon([(315, 240), (305, 280), (325, 280)], fill=(200, 175, 155))
    # Mouth
    draw.arc([285, 290, 355, 340], 0, 180, fill=(180, 80, 80), width=2)
    # Eyebrows
    draw.arc([265, 160, 315, 185], 180, 360, fill=(100, 70, 50), width=2)
    draw.arc([325, 160, 375, 185], 180, 360, fill=(100, 70, 50), width=2)
    img.save(output_dir / "test_face_basic.jpg", "JPEG", quality=90)

    # 2. No face (landscape)
    img = Image.new("RGB", (640, 480), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Sky gradient effect
    for y in range(480):
        r = int(135 + (y / 480) * 50)
        g = int(206 - (y / 480) * 100)
        b = int(235 - (y / 480) * 50)
        draw.line([(0, y), (640, y)], fill=(r, g, b))
    # Mountains
    draw.polygon([(0, 350), (160, 200), (320, 350)], fill=(100, 100, 100))
    draw.polygon([(200, 350), (400, 150), (600, 350)], fill=(80, 80, 80))
    # Ground
    draw.rectangle([0, 350, 640, 480], fill=(34, 139, 34))
    img.save(output_dir / "test_no_face.jpg", "JPEG", quality=90)

    # 3. Tiny image
    img = Image.new("RGB", (32, 32), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 4, 24, 28], fill=(220, 195, 175))
    img.save(output_dir / "test_tiny.jpg", "JPEG")

    # 4. Large image (1920x1080)
    img = Image.new("RGB", (1920, 1080), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)
    # Large face
    draw.ellipse(
        [710, 140, 1210, 940],
        fill=(220, 195, 175),
        outline=(180, 160, 140),
        width=3,
    )
    draw.ellipse([830, 350, 910, 420], fill=(255, 255, 255))
    draw.ellipse([850, 370, 890, 400], fill=(60, 40, 20))
    draw.ellipse([1010, 350, 1090, 420], fill=(255, 255, 255))
    draw.ellipse([1030, 370, 1070, 400], fill=(60, 40, 20))
    draw.arc([870, 550, 1050, 680], 0, 180, fill=(180, 80, 80), width=3)
    img.save(output_dir / "test_large.jpg", "JPEG", quality=95)

    print(f"Created 4 static test images in {output_dir}")


if __name__ == "__main__":
    create_test_images()
