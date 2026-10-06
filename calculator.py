"""
calculator.py - Deterministic Construction Concrete Quantity Calculator Module
Part of BuildMate AI - Intelligent Construction Site Assistant
"""

def calculate_concrete_volume(quantity: float, length: float, width: float, depth: float) -> float:
    """
    Computes nominal concrete volume for structural components.
    Formula: Volume (m³) = Quantity × Length (m) × Width (m) × Depth/Thickness (m)
    """
    if quantity <= 0 or length <= 0 or width <= 0 or depth <= 0:
        raise ValueError("All dimensions and quantity must be greater than zero.")
    
    volume = quantity * length * width * depth
    return round(volume, 3)

def calculate_structural_element(element_type: str, quantity: float, length: float, width: float, depth: float) -> dict:
    """
    Evaluates volumetric quantities and returns structured site audit details.
    """
    vol = calculate_concrete_volume(quantity, length, width, depth)
    return {
        "element": element_type,
        "quantity": quantity,
        "length_m": length,
        "width_m": width,
        "depth_m": depth,
        "volume_m3": vol,
        "unit": "cubic metres (m³)"
    }
