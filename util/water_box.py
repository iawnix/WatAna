from typing import List, Tuple
from rich import print as rprint

global NA, M_PER_WATER, RHO_PER_WATER, INO_NUM_MAP
M_PER_WATER = 18.02                                       # g/mol
NA = 6.02214076e23                                        # mol^-1
RHO_PER_WATER = 1.0                                       # 1 g/cm^3
V_PER_WATER = M_PER_WATER / (NA * RHO_PER_WATER)* 1e21    # cm -> nm

INO_NUM_MAP = {
        "NACL": (("Na", 1), ("Cl", 1))
       ,"NA":   (("Na", 1))
       ,"CL":   (("Cl", 1))
       ,"KCL":  (("K", 1), ("Cl", 1))
       ,"K":    (("K", 1))}



def water_to_box(n_water: int, print_info: bool = False) -> Tuple[float, float]:

    V = n_water * V_PER_WATER
    L = V ** (1/3)
    if print_info:
        rprint("The num of H2O is {}, the L is {:.4f} nm, and the V is {:.4} nm^3".format(n_water, L, V))

    return (L, V)

def box_to_water(L_s: List, print_info: bool = False) -> int:
    
    assert len(L_s) == 3, rprint("Error, you must privide the box xyz info!, L_s: {}.".format(L_s))

    V_box = L_s[0] * L_s[1] * L_s[2]
    n_water = int(V_box / V_PER_WATER)
    
    if print_info:
        rprint("The box (x: {:.4f} nm, y: {:.4f} nm, z: {:.4f} nm) can contain {} H2O".format(L_s[0], L_s[1], L_s[2], n_water))
    
    return n_water



def box_to_water_ino(L_s: List, ino_s: str, concentration: float, print_info: bool = False) -> List[Tuple[float, int]]:
    
    assert len(L_s) == 3, rprint("Error, you must privide the box xyz info!, L_s: {}.".format(L_s))

    V_box = L_s[0] * L_s[1] * L_s[2]
    V_box *= 1e-24                          # nm^3 -> L
    n_salt = int(concentration * V_box * NA)
    
    ino_s = ino_s.upper()
    var = INO_NUM_MAP[ino_s]
    ino_out = []
    for i in var:
        ino_out.append((i[0], i[1]*n_salt))
    
    if print_info:
        rprint("The box (x: {:.4f} nm, y: {:.4f} nm, z: {:.4f} nm) can contain {}[{:.4f}]: {}".format(L_s[0], L_s[1], L_s[2], ino_s, concentration, ino_out))
    return ino_out

if __name__ == "__main__":
    water_to_box(216, True)
    water_to_box(512, True)
    box_to_water([5.0, 5.0, 5.0], True)
    box_to_water_ino([5.0, 5.0, 5.0], "Nacl", 0.15, True)
