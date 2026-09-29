def calculate_carbon_stock(agb_kg):
    carbon_fraction = 0.47
    return agb_kg * carbon_fraction


def calculate_co2_equivalent(carbon_kg):
    co2_molecular_weight = 44.01
    carbon_molecular_weight = 12.01
    ratio = co2_molecular_weight / carbon_molecular_weight
    return carbon_kg * ratio
