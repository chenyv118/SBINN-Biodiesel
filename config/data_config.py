import scipy.io

# Const data
rhoOil = 920  # density oil [g/L]
rmmOil = 880  # RMM Oil [g/mol]
MH2O = 18.02  # RMM water [g/mol]
rhoAlco = 792  # density Methanol  % [g/L]
rmmAlco = 32.04  # RMM Alco [g/mol]
Dp = 5.88e-6  # [m] Sauter mean diameter of dispersed phase
RMMBioCat = 34000  # RMM Enzyme [g/mol]
rmmFAME = 294.9  # RMM Biodisel [g/mol] Fatty acid methyl esters (biodiesel)
rmmMAG = 355.0  # Monoglycerides [mass %]
rmmDAG = 617.9  # Diglycerides [mass %]
rmmTAG = 880.8  # Triglycerides [mass %]
rmmFFA = 280.9  # Free fatty acids  [mass %]

# Experiment data
data = scipy.io.loadmat("config/BD2J.mat").get("num").tolist()
Moil = data[0][0]  # Mass of Oil [g]
Vo = data[23][0]  # Inital total volume L
vH2O = data[10][0] / 1000  # volume water L
Mao = data[21][0]  # Inital mass added alcohol  [g]
Vao = Mao / rhoAlco  # Inital volume added alcohol  [L]
CoAlco = rhoAlco / rmmAlco  # Concentration alcohol in feed mol/L
Fa = data[27][0: 2]
Fa = [i * 60 for i in Fa]  # Alcohol flow rate of methanol  .... L/min
Va_total = data[6][0]  # Total amount of alcohol added [L]
Vreactf = data[23][-1]  # Final reactor volume
Vpo = vH2O  # Initial polar volume [L]
ActCat = 0.07  # % of active catalyst
Enzyme = ActCat * data[1][0] / Vo / RMMBioCat  # biocatalyst conventration mol/L
mH2O = data[10][0] / Vo / 18  # Inital concentration water mol/L
Alco = data[20][0] / Vo  # Initial conc in polar volume  mol/L

# const Parameter
af = 6 / Dp  # Free specific interfacial area [mol/m3], interfacial area m2/m3，
Ae = 2.98e7  # enzyme coverage
