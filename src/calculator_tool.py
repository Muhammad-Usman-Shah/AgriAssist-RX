
"""
AgriAssist-RX
Phase 4, Step 8: Agricultural Calculator

Supported calculations:
1. Fertilizer quantity
2. Fertilizer cost
3. Seed requirement
4. Acres to hectares
5. Irrigation water volume

All calculations use user-provided values.

No Gemini API calls.
No external packages required.
"""

from decimal import Decimal, InvalidOperation


# ==========================================
# STEP 1: Validate numeric inputs
# ==========================================

def positive_number(value, name):

    try:
        number = Decimal(str(value))

    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(
            f"{name} must be a valid number."
        )

    if not number.is_finite() or number <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return number


def nonnegative_number(value, name):

    try:
        number = Decimal(str(value))

    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(
            f"{name} must be a valid number."
        )

    if not number.is_finite() or number < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return number


# ==========================================
# STEP 2: Fertilizer quantity
# ==========================================

def fertilizer_quantity(area_acres, bags_per_acre):

    area = positive_number(area_acres, "Area")

    rate = nonnegative_number(
        bags_per_acre,
        "Fertilizer rate"
    )

    total_bags = area * rate

    return {
        "calculation": "fertilizer_quantity",
        "total_bags": total_bags,
        "unit": "bags"
    }


# ==========================================
# STEP 3: Fertilizer cost
# ==========================================

def fertilizer_cost(number_of_bags, price_per_bag):

    bags = nonnegative_number(
        number_of_bags,
        "Number of bags"
    )

    price = nonnegative_number(
        price_per_bag,
        "Price per bag"
    )

    total_cost = bags * price

    return {
        "calculation": "fertilizer_cost",
        "total_cost_pkr": total_cost,
        "unit": "PKR"
    }


# ==========================================
# STEP 4: Seed requirement
# ==========================================

def seed_requirement(area_acres, seed_rate_kg_acre):

    area = positive_number(area_acres, "Area")

    rate = nonnegative_number(
        seed_rate_kg_acre,
        "Seed rate"
    )

    total_seed = area * rate

    return {
        "calculation": "seed_requirement",
        "total_seed_kg": total_seed,
        "unit": "kg"
    }


# ==========================================
# STEP 5: Area conversion
# ==========================================

def acres_to_hectares(area_acres):

    area = nonnegative_number(
        area_acres,
        "Area"
    )

    # International acre conversion.
    hectares = area * Decimal("0.40468564224")

    return {
        "calculation": "area_conversion",
        "hectares": hectares,
        "unit": "hectares"
    }


# ==========================================
# STEP 6: Irrigation volume
# ==========================================

def irrigation_volume(area_acres, depth_mm):

    area = positive_number(area_acres, "Area")

    depth = nonnegative_number(
        depth_mm,
        "Water depth"
    )

    # 1 acre = 4046.8564224 square metres.
    # 1 mm = 0.001 metre.
    area_m2 = area * Decimal("4046.8564224")

    depth_m = depth / Decimal("1000")

    volume_m3 = area_m2 * depth_m

    return {
        "calculation": "irrigation_volume",
        "volume_m3": volume_m3,
        "unit": "cubic metres"
    }


# ==========================================
# STEP 7: Interactive calculator
# ==========================================

def show_menu():

    print("\n" + "=" * 55)
    print("AGRIASSIST-RX: AGRICULTURAL CALCULATOR")
    print("=" * 55)

    print("1. Fertilizer quantity")
    print("2. Fertilizer cost")
    print("3. Seed requirement")
    print("4. Acres to hectares")
    print("5. Irrigation water volume")
    print("0. Exit")


def main():

    while True:

        show_menu()

        choice = input(
            "\nChoose an option (0-5): "
        ).strip()

        if choice == "0":
            print("\nGoodbye!")
            break

        try:

            if choice == "1":

                area = input("Area (acres): ")

                rate = input(
                    "Fertilizer rate (bags/acre): "
                )

                result = fertilizer_quantity(
                    area, rate
                )

                print(
                    "\nTotal fertilizer:",
                    result["total_bags"],
                    "bags"
                )

            elif choice == "2":

                bags = input(
                    "Number of fertilizer bags: "
                )

                price = input(
                    "Price per bag (PKR): "
                )

                result = fertilizer_cost(
                    bags, price
                )

                print(
                    "\nTotal cost: PKR",
                    result["total_cost_pkr"]
                )

            elif choice == "3":

                area = input("Area (acres): ")

                rate = input(
                    "Seed rate (kg/acre): "
                )

                result = seed_requirement(
                    area, rate
                )

                print(
                    "\nTotal seed required:",
                    result["total_seed_kg"],
                    "kg"
                )

            elif choice == "4":

                area = input("Area (acres): ")

                result = acres_to_hectares(area)

                print(
                    "\nArea:",
                    round(result["hectares"], 4),
                    "hectares"
                )

            elif choice == "5":

                area = input("Area (acres): ")

                depth = input(
                    "Irrigation depth (mm): "
                )

                result = irrigation_volume(
                    area, depth
                )

                print(
                    "\nIrrigation water volume:",
                    round(result["volume_m3"], 2),
                    "m³"
                )

            else:

                print(
                    "\nInvalid option. Choose 0-5."
                )

        except ValueError as error:

            print("\nInput error:", error)


if __name__ == "__main__":
    main()
