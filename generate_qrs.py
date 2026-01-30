# generate_qrs.py

import qrcode
import os

# Try to import styled QR code modules (optional)
try:
    from qrcode.image.styledmod import StyledPilImage
    from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
    from qrcode.image.styles.colorfill import RadialGradiantColorMask
    STYLED_QR_AVAILABLE = True
except ImportError:
    STYLED_QR_AVAILABLE = False
    print("Note: Styled QR codes not available, using standard QR codes.")
    print("      Install qrcode[pil] with: pip install 'qrcode[pil]'")

# Updated URL for the new FastAPI collector endpoint
# Using 127.0.0.1:8000 (FastAPI default) with the /collector route
COLLECTOR_APP_URL_TEMPLATE = "http://127.0.0.1:8000/collector?bin_id={}"

# For production, you might want to use your actual domain:
# COLLECTOR_APP_URL_TEMPLATE = "https://yourdomain.com/collector?bin_id={}"

# Your bin IDs
BINS = ["bin-A", "bin-B", "bin-C", "bin-D"]

# Create a directory to store the QR codes
qr_dir = "bin_qrs"
os.makedirs(qr_dir, exist_ok=True)

# Enhanced QR code generation with styling
def generate_styled_qr(data, bin_id):
    """Generate a styled QR code with rounded corners and gradient"""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create styled image with rounded modules and gradient
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=RoundedModuleDrawer(),
            color_mask=RadialGradiantColorMask(
                back_color=(255, 255, 255),
                center_color=(102, 126, 234),  # Purple
                edge_color=(118, 75, 162)      # Darker purple
            )
        )
        return img
    except (ImportError, AttributeError):
        # Fallback if styled modules are not available
        raise

print("Generating QR codes for WasteWise bins...")
print(f"Using URL template: {COLLECTOR_APP_URL_TEMPLATE}")
print("-" * 50)

for bin_id in BINS:
    # Construct the unique URL for each bin
    bin_url = COLLECTOR_APP_URL_TEMPLATE.format(bin_id)
    
    if STYLED_QR_AVAILABLE:
        try:
            # Try to generate styled QR code
            qr_img = generate_styled_qr(bin_url, bin_id)
        except Exception as e:
            # Fallback to simple QR code if styling fails
            print(f"Warning: Could not generate styled QR for {bin_id}, using simple version: {e}")
            qr_img = qrcode.make(bin_url)
    else:
        # Use simple QR code
        qr_img = qrcode.make(bin_url)
    
    # Save the QR code as a PNG file
    file_path = os.path.join(qr_dir, f"qr_code_{bin_id}.png")
    qr_img.save(file_path)
    print(f"[OK] Generated QR code for {bin_id}")
    print(f"     URL: {bin_url}")
    print(f"     Saved to: {file_path}")
    print()

print("-" * 50)
print("QR code generation complete!")
print(f"All QR codes saved in: {os.path.abspath(qr_dir)}")
print("\nNote: Make sure your FastAPI server is running on http://127.0.0.1:8000")
print("      or update the COLLECTOR_APP_URL_TEMPLATE for your production domain.")