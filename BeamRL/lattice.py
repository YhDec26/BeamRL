from ocelot import (
    Aperture,
    Cavity,
    Drift,
    Hcor,
    Marker,
    Monitor,
    Quadrupole,
    SBend,
    Solenoid,
    TDCavity,
    Vcor,
)

# from cheetah import BPM, Drift, HorizontalCorrector, Segment, Dipole, Screen, Solenoid, Quadrupole, Aperture
# import cheetah
# import torch

from cheetah import BPM

# Drifts
# drift_0.5 = Aperture(100)
drift_1 = Drift(l=0.001, eid="Drift_1")
drift_2 = Drift(l=0.024, eid="Drift_2")
drift_3 = Drift(l=0.025, eid="Drift_3")
drift_4 = Drift(l=0.025, eid="Drift_4")
drift_5 = Drift(l=0.025, eid="Drift_5")
drift_6 = Drift(l=0.025, eid="Drift_6")
drift_7 = Drift(l=0.025, eid="Drift_7")
drift_8 = Drift(l=0.025, eid="Drift_8")
drift_9 = Drift(l=0.025, eid="Drift_9")
drift_10 = Drift(l=0.025, eid="Drift_10")
drift_11 = Drift(l=0.090, eid="Drift_11")
#drift_11.5 = S1
drift_12 = Drift(l=0.20, eid="Drift_12")
drift_13 = Drift(l=0.001, eid="Drift_13")
drift_14 = Drift(l=0.049, eid="Drift_14")
drift_15 = Drift(l=0.490, eid="Drift_15")
drift_16 = Drift(l=0.010, eid="Drift_16")
drift_17 = Drift(l=0.240, eid="Drift_17")
#drift_17.5 = S2
drift_18 = Drift(l=0.240, eid="Drift_18")
drift_19 = Drift(l=0.240, eid="Drift_19")
drift_20 = Drift(l=0.240, eid="Drift_20")
#drift_20.5 = S3 + ion chamber
drift_21 = Drift(l=0.0295, eid="Drift_21")
drift_22 = Drift(l=0.0295, eid="Drift_22")
drift_23 = Drift(l=0.0295, eid="Drift_23")
drift_24 = Drift(l=0.0295, eid="Drift_24")
drift_25 = Drift(l=0.0295, eid="Drift_25")
drift_26 = Drift(l=0.0295, eid="Drift_26")
drift_27 = Drift(l=0.0295, eid="Drift_27")
drift_28 = Drift(l=0.0295, eid="Drift_28")
drift_29 = Drift(l=0.0295, eid="Drift_29")
drift_30 = Drift(l=0.0295, eid="Drift_30")
drift_31 = Drift(l=0.010, eid="Drift_31")
drift_32 = Drift(l=0.295, eid="Drift_32")
#drift_32.5 = Q1
drift_33 = Drift(l=0.300, eid="Drift_33")
#drift_33.5 = Q2
drift_34 = Drift(l=0.400, eid="Drift_34")
#drift_34.5 = Aperture(36) + B1 + BPM1
drift_35 = Drift(l=0.400, eid="Drift_35")
#drift_35.5 = Q3
drift_36 = Drift(l=0.220, eid="Drift_36")
drift_37 = Drift(l=0.001, eid="Drift_37")
drift_38 = Drift(l=0.270, eid="Drift_38")
#drift_38.5 = Q4
drift_39 = Drift(l=0.400, eid="Drift_39")
#drift_39.5 = B2
drift_40 = Drift(l=0.400, eid="Drift_40")
#drift_40.5 = BPM2
drift_41 = Drift(l=0.145, eid="Drift_41")
#drift_41.5 = Q5
drift_42 = Drift(l=0.200, eid="Drift_42")
#drift_42.5 = Q6
drift_43 = Drift(l=0.500, eid="Drift_43")
#drift_43.5 = QTx
drift_44 = Drift(l=0.200, eid="Drift_44")
#drift_44.5 = Q7
drift_45 = Drift(l=0.200, eid="Drift_45")
#drift_45.5 = Q8
drift_46 = Drift(l=0.500, eid="Drift_46")
#drift_46.5 = QTy
drift_47 = Drift(l=0.200, eid="Drift_47")
#drift_47.5 = Qj6
drift_48 = Drift(l=0.200, eid="Drift_48")
#drift_48.5 = Qj7
drift_49 = Drift(l=0.200, eid="Drift_49")
#drift_49.5 = Qj8
drift_50 = Drift(l=0.010, eid="Drift_50")
drift_51 = Drift(l=0.340, eid="Drift_51")
drift_52 = Drift(l=0.350, eid="Drift_52")
#drift_52.5 = ion_chamber_2
drift_53 = Drift(l=0.350, eid="Drift_53")
drift_54 = Drift(l=0.350, eid="Drift_54")
drift_55 = Drift(l=0.350, eid="Drift_55")
drift_56 = Drift(l=0.350, eid="Drift_56")
drift_57 = Drift(l=0.350, eid="Drift_57")
drift_58 = Drift(l=0.350, eid="Drift_58")
drift_59 = Drift(l=0.350, eid="Drift_59")
drift_60 = Drift(l=0.350, eid="Drift_60")
#drift_60.5 = Aperture(50)


# Quadrupoles
Q1 = Quadrupole(l=0.150, eid="Q1")
Q2 = Quadrupole(l=0.150, eid="Q2")
Q3 = Quadrupole(l=0.150, eid="Q3")
Q4 = Quadrupole(l=0.150, eid="Q4")
Q5 = Quadrupole(l=0.150, eid="Q5")
Q6 = Quadrupole(l=0.150, eid="Q6")
QTx = Quadrupole(l=0.400, eid="QTx")
Q7 = Quadrupole(l=0.350, eid="Q7")
Q8 = Quadrupole(l=0.300, eid="Q8")
QTy = Quadrupole(l=0.300, eid="QTy")
Qj6 = Quadrupole(l=0.300, eid="Qj6")
Qj7 = Quadrupole(l=0.350, eid="Qj7")
Qj8 = Quadrupole(l=0.300, eid="Qj8")

# SBends(aperture=30)
B1 = SBend(
    l=0.7854,
    angle=-0.78539816339744831,
    eid="B1",
)

B2 = SBend(
    l=0.7854,
    angle=-0.78539816339744831,
    eid="B2",
)

# Hcors(behind Q2, Q4, Q6)
H1 = Hcor(
    l=0.001,
    eid="H1",
)
H2 = Hcor(
    l=0.001,
    eid="H2",
)
H3 = Hcor(
    l=0.001,
    eid="H3",
)


# arlimcxg1a = Hcor(l=5e-05, eid="ARLIMCXG1")

# Solenoids
S1 = Solenoid(l=0.800, eid="S1")
S2 = Solenoid(l=0.800, eid="S2")
S3 = Solenoid(l=0.800, eid="S3")

# Monitors
BSC1 = Monitor(eid="BSC1")
BPM1 = Monitor(eid="BPM1")
BPM2 = Monitor(eid="BPM2")
BSC2 = Monitor(eid="BSC2")

# Markers
# arlisolg1 = Marker(eid="ARLISOLG1")

# Apertures(behind elements)
aperture_1 = Aperture(eid="aperture_1", xmax=0.100, ymax=0.100)
aperture_2 = Aperture(eid="aperture_2", xmax=0.050, ymax=0.050)
aperture_3 = Aperture(eid="aperture_3", xmax=0.050, ymax=0.050)
aperture_4 = Aperture(eid="aperture_4", xmax=0.050, ymax=0.050)
aperture_5 = Aperture(eid="aperture_5", xmax=0.050, ymax=0.050)
aperture_6 = Aperture(eid="aperture_6", xmax=0.050, ymax=0.050)
aperture_7 = Aperture(eid="aperture_7", xmax=0.050, ymax=0.050)
aperture_8 = Aperture(eid="aperture_8", xmax=0.050, ymax=0.050)
aperture_9 = Aperture(eid="aperture_9", xmax=0.050, ymax=0.050)
aperture_10 = Aperture(eid="aperture_10", xmax=0.050, ymax=0.050)
aperture_11 = Aperture(eid="aperture_11", xmax=0.050, ymax=0.050)
aperture_12 = Aperture(eid="aperture_12", xmax=0.050, ymax=0.050)
# S1
aperture_13 = Aperture(eid="aperture_13", xmax=0.080, ymax=0.080)
aperture_14 = Aperture(eid="aperture_14", xmax=0.060, ymax=0.060)
aperture_15 = Aperture(eid="aperture_15", xmax=0.050, ymax=0.050)
aperture_16 = Aperture(eid="aperture_16", xmax=0.050, ymax=0.050)
aperture_17 = Aperture(eid="aperture_17", xmax=0.060, ymax=0.060)
aperture_18 = Aperture(eid="aperture_18", xmax=0.060, ymax=0.060)
aperture_19 = Aperture(eid="aperture_19", xmax=0.060, ymax=0.060)
# S2
aperture_20 = Aperture(eid="aperture_20", xmax=0.060, ymax=0.060)
aperture_21 = Aperture(eid="aperture_21", xmax=0.060, ymax=0.060)
aperture_22 = Aperture(eid="aperture_22", xmax=0.030, ymax=0.030)
aperture_23 = Aperture(eid="aperture_23", xmax=0.030, ymax=0.030)
# S3
aperture_24 = Aperture(eid="aperture_24", xmax=0.030, ymax=0.030)
aperture_25 = Aperture(eid="aperture_25", xmax=0.030, ymax=0.030)
aperture_26 = Aperture(eid="aperture_26", xmax=0.030, ymax=0.030)
aperture_27 = Aperture(eid="aperture_27", xmax=0.030, ymax=0.030)
aperture_28 = Aperture(eid="aperture_28", xmax=0.030, ymax=0.030)
aperture_29 = Aperture(eid="aperture_29", xmax=0.030, ymax=0.030)
aperture_30 = Aperture(eid="aperture_30", xmax=0.030, ymax=0.030)
aperture_31 = Aperture(eid="aperture_31", xmax=0.030, ymax=0.030)
aperture_32 = Aperture(eid="aperture_32", xmax=0.030, ymax=0.030)
aperture_33 = Aperture(eid="aperture_33", xmax=0.030, ymax=0.030)
aperture_34 = Aperture(eid="aperture_34", xmax=0.030, ymax=0.030)
aperture_35 = Aperture(eid="aperture_35", xmax=0.030, ymax=0.030)
aperture_36 = Aperture(eid="aperture_36", xmax=0.030, ymax=0.030)
# Q1
aperture_37 = Aperture(eid="aperture_37", xmax=0.036, ymax=0.036)
aperture_38 = Aperture(eid="aperture_38", xmax=0.036, ymax=0.036)
# Q2
aperture_39 = Aperture(eid="aperture_39", xmax=0.036, ymax=0.036)
aperture_40 = Aperture(eid="aperture_40", xmax=0.036, ymax=0.036)
aperture_41 = Aperture(eid="aperture_41", xmax=0.036, ymax=0.036)
# B1
aperture_42 = Aperture(eid="aperture_42", xmax=0.030, ymax=0.030)
aperture_43 = Aperture(eid="aperture_43", xmax=0.036, ymax=0.036)
# Q3
aperture_44 = Aperture(eid="aperture_44", xmax=0.036, ymax=0.036)
aperture_45 = Aperture(eid="aperture_45", xmax=0.036, ymax=0.036)
aperture_46 = Aperture(eid="aperture_46", xmax=0.036, ymax=0.036)
aperture_47 = Aperture(eid="aperture_47", xmax=0.036, ymax=0.036)
# Q4
aperture_48 = Aperture(eid="aperture_48", xmax=0.036, ymax=0.036)
aperture_49 = Aperture(eid="aperture_49", xmax=0.036, ymax=0.036)
# B2
aperture_50 = Aperture(eid="aperture_50", xmax=0.050, ymax=0.050)
aperture_51 = Aperture(eid="aperture_51", xmax=0.060, ymax=0.060)
aperture_52 = Aperture(eid="aperture_52", xmax=0.036, ymax=0.036)
# Q5
aperture_53 = Aperture(eid="aperture_53", xmax=0.060, ymax=0.060)
aperture_54 = Aperture(eid="aperture_54", xmax=0.060, ymax=0.060)
# Q6
aperture_55 = Aperture(eid="aperture_55", xmax=0.060, ymax=0.060)
aperture_56 = Aperture(eid="aperture_56", xmax=0.060, ymax=0.060)
# QTx
aperture_57 = Aperture(eid="aperture_57", xmax=0.050, ymax=0.050)
aperture_58 = Aperture(eid="aperture_58", xmax=0.060, ymax=0.060)
# Q7
aperture_59 = Aperture(eid="aperture_59", xmax=0.060, ymax=0.060)
aperture_60 = Aperture(eid="aperture_60", xmax=0.050, ymax=0.050)
# Q8
aperture_61 = Aperture(eid="aperture_61", xmax=0.060, ymax=0.060)
aperture_62 = Aperture(eid="aperture_62", xmax=0.050, ymax=0.050)
# QTy
aperture_63 = Aperture(eid="aperture_63", xmax=0.050, ymax=0.050)
aperture_64 = Aperture(eid="aperture_64", xmax=0.050, ymax=0.050)
# Qj6
aperture_65 = Aperture(eid="aperture_65", xmax=0.060, ymax=0.060)
aperture_66 = Aperture(eid="aperture_66", xmax=0.050, ymax=0.050)
# Qj7
aperture_67 = Aperture(eid="aperture_67", xmax=0.060, ymax=0.060)
aperture_68 = Aperture(eid="aperture_68", xmax=0.050, ymax=0.050)
# Qj8
aperture_69 = Aperture(eid="aperture_69", xmax=0.060, ymax=0.060)
aperture_70 = Aperture(eid="aperture_70", xmax=0.050, ymax=0.050)
aperture_71 = Aperture(eid="aperture_71", xmax=0.050, ymax=0.050)
aperture_72 = Aperture(eid="aperture_72", xmax=0.050, ymax=0.050)
aperture_73 = Aperture(eid="aperture_73", xmax=0.050, ymax=0.050)
aperture_74 = Aperture(eid="aperture_74", xmax=0.050, ymax=0.050)
aperture_75 = Aperture(eid="aperture_75", xmax=0.050, ymax=0.050)
aperture_76 = Aperture(eid="aperture_76", xmax=0.050, ymax=0.050)
aperture_77 = Aperture(eid="aperture_77", xmax=0.050, ymax=0.050)
aperture_78 = Aperture(eid="aperture_78", xmax=0.050, ymax=0.050)
aperture_79 = Aperture(eid="aperture_79", xmax=0.050, ymax=0.050)
aperture_80 = Aperture(eid="aperture_80", xmax=0.050, ymax=0.050)
aperture_81 = Aperture(eid="aperture_81", xmax=0.050, ymax=0.050)

# Lattice
cell = (
    aperture_1,drift_1,
    aperture_2,drift_2,
    aperture_3,drift_3,
    aperture_4,drift_4,
    aperture_5,drift_5,
    aperture_6,drift_6,
    aperture_7,drift_7,
    aperture_8, drift_8,
    aperture_9, drift_9,
    aperture_10, drift_10,
    aperture_11,drift_11,
    aperture_12,S1,
    aperture_13,drift_12,
    aperture_14,drift_13,
    aperture_15,drift_14,
    aperture_16,drift_15,
    aperture_17,drift_16,
    aperture_18,drift_17,
    aperture_19,S2,
    aperture_20,drift_18,
    aperture_21,drift_19,
    aperture_22,drift_20,
    aperture_23,S3,BSC1,
    aperture_24,drift_21,
    aperture_25,drift_22,
    aperture_26,drift_23,
    aperture_27,drift_24,
    aperture_28,drift_25,
    aperture_29,drift_26,
    aperture_30,drift_27,
    aperture_31,drift_28,
    aperture_32,drift_29,
    aperture_33,drift_30,
    aperture_34,drift_31,
    aperture_35,drift_32,
    aperture_36,Q1,
    aperture_37,drift_33,
    aperture_38,Q2,H1,
    aperture_39,drift_34,
    aperture_40,B1,BPM1,aperture_41,
    aperture_42,drift_35,
    aperture_43,Q3,
    aperture_44,drift_36,
    aperture_45,drift_37,
    aperture_46,drift_38,
    aperture_47,Q4,H2,
    aperture_48,drift_39,
    aperture_49,B2,
    aperture_50,drift_40,BPM2,
    aperture_51,drift_41,
    aperture_52,Q5,
    aperture_53,drift_42,
    aperture_54,Q6,H3,
    aperture_55,drift_43,
    aperture_56,QTx,
    aperture_57,drift_44,
    aperture_58,Q7,
    aperture_59,drift_45,
    aperture_60,Q8,
    aperture_61,drift_46,
    aperture_62,QTy,
    aperture_63,drift_47,
    aperture_64,Qj6,
    aperture_65,drift_48,
    aperture_66,Qj7,
    aperture_67,drift_49,
    aperture_68,Qj8,
    aperture_69,drift_50,
    aperture_70,drift_51,
    aperture_71,drift_52,BSC2,
    aperture_72,drift_53,
    aperture_73,drift_54,
    aperture_74,drift_55,
    aperture_75,drift_56,
    aperture_76,drift_57,
    aperture_77,drift_58,
    aperture_78,drift_59,
    aperture_79,drift_60,
    aperture_80,aperture_81
)
