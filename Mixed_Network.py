import numpy as np
from scipy.integrate import solve_ivp

class NetworkEquations:
    def __init__(self):
        self.array_of_elements = None
        self.n = None
        self.out_of_network_arguments = None
        self.Nodes()
        self.networkparameters()
        self.create_ode_system()
        self.solve()

    def Nodes(self):
        self.array_of_elements = np.array(['IFNGR', 'CSF2RA', 'IL1R', 'TLR4', 'FCGR', 'IL4RA', 'IL10R', 'STAT1', 'MSTAT5', 'MNFKB', 'PPARG', 'STAT6', 'JMJD3', 'STAT3', 'SOCS3', 'IRF3', 'ERK', 'KLF4', 'SOCS1', 'IRF4', 'IL1BOUT', 'IL12OUT', 'TAK123', 'JNKMAPK', 'ERK12', 'MSK12', 'GSK3B', 'IL10OUT', 'AKT1', 'AKT3', 'TLR2', 'TLR3', 'TLR7', 'TLR8', 'TLR9', 'MYD88', 'IRAK4', 'IRAK12', 'TRAF6', 'TAB23TAK', 'MKK36', 'P38', 'CREB1', 'MKK47', 'JNK', 'CJUN', 'MKK12', 'CFOS', 'MAP1', 'NEMOIKKB', 'RIP1', 'TLR4END', 'TRIF', 'TRAF3', 'TBK1IKKI', 'IKKA', 'IRF7', 'IL6OUT', 'IL18OUT', 'IL33OUT', 'IFNABOUT', 'PHAGOCYTOSIS', 'PHAGOSOME', 'PROCESSING', 'MHC2', 'CD8086', 'ITAM', 'SYK', 'CARD9', 'VAV', 'RAC', 'CDC42', 'WASP', 'WAVE', 'MCA', 'LYN', 'FYN', 'DAP12', 'MPI3K', 'MPLC', 'MAKT', 'MPKC', 'CD11BCD18', 'HCK', 'FGR', 'FCAR', 'MMTOR', 'MMTORC1', 'MMTORC2', 'MLKB1', 'MAMPK', 'MGlycolysis', 'MOXPHOS', 'MAMPATPratio', 'MHIF1A', 'RIG1', 'MAVS', 'TRADD', 'NEMOIKKAB', 'NEMOTBK1IKKE', 'TNFR1', 'TRAF2', 'TNFAOUT', 'DECTIN1', 'SRC', 'DECTIN2', 'MR', 'CLEC10A', 'MINCLE', 'TCR', 'CD28', 'AP1', 'CD25', 'IL2G', 'IL2E', 'MTOR', 'ZAP70', 'STAT5', 'NFAT', 'NFKB', 'AKT', 'CTLA4', 'CTLA4DIM', 'BCL2', 'NDRG1', 'DAG', 'SOS', 'RASGTPR', 'LCK', 'PDK1', 'LAT', 'PLC', 'PI3K', 'PIP2', 'PIP3', 'IP3', 'CA', 'PKC', 'TBET', 'IFNG', 'GATA3', 'IL4', 'FOXP3', 'IL10', 'TGFB', 'RORGT', 'IL21', 'IL17', 'BCL6', 'IL9', 'CD40L', 'MTORC1', 'MTORC2', 'LKB1', 'AMPK', 'Glycolysis', 'GLUTAMINOLISIS', 'AKG', 'OXPHOS', 'AMPATPratio', 'HIF1A'])
        self.n = len(self.array_of_elements)
        return self.array_of_elements

    def networkparameters(self):
        self.out_of_network_arguments = np.array(['b', 'LP', 'DSRNA', 'SSRNA', 'CPGDNA', 'IFNGE', 'IFNGL', 'MHC2E', 'CD8086E', 'GMCSFE', 'IL1BE', 'LPSE', 'IL4E', 'IL4L', 'TGFBE', 'IL10E', 'IL10L', 'IL12E', 'IC3B', 'IGGC', 'IGAC', 'IL6E', 'IL21E', 'IL18E', 'IL33E', 'MGLC', 'GLC', 'CITDSRNA', 'CITSSRNA', 'TNFA', 'CIAP', 'GLN', 'MFA', 'FA', 'TRP', 'MO2', 'O2', 'METF', 'RAPA', 'PRED', 'CS', 'TCD8086', 'TMHC2'])
        return self.out_of_network_arguments

    def create_ode_system(self, af=None, decay_rates=None, threshold_values=None):
        def act_f(a_function=None, x=None, b=None, u=None):
            if a_function is None:
                a_function = 'sigmoid'
            if a_function == 'sigmoid':
                return 1 / (1 + np.exp(-b*(x-u)))
            elif a_function == 'tanh':
                return (1+np.tanh(-b*(x-u)))/2
            elif a_function == 'relu':
                return np.maximum(0,-b*(x-u))

        def system_ode(t, q, *p):
            w = np.zeros(len(q))
            w[0] = p[5]
            w[1] = p[9]
            w[2] = (p[10]+q[20]-p[10]*q[20])
            w[3] = p[11]
            w[4] = ((p[19]+p[19]*p[10]-p[19]*p[19]*p[10])+q[108]-(p[19]+p[19]*p[10]-p[19]*p[19]*p[10])*q[108])
            w[5] = p[12]
            w[6] = (p[15]+q[27]-p[15]*q[27])
            w[7] = q[0]*(1-(q[18]+q[13]-q[18]*q[13]))
            w[8] = q[1]*(1-((q[13]+q[19]-q[13]*q[19])+q[18]-(q[13]+q[19]-q[13]*q[19])*q[18]))
            w[9] = (((((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])+q[81]-(((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])*q[81])+q[50]*q[101]*p[30]-((((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])+q[81]-(((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])+q[80]-((q[2]+q[49]-q[2]*q[49])+q[68]-(q[2]+q[49]-q[2]*q[49])*q[68])*q[80])*q[81])*q[50]*q[101]*p[30])*(1-(((q[13]+q[10]-q[13]*q[10])+q[17]-(q[13]+q[10]-q[13]*q[10])*q[17])+q[42]-((q[13]+q[10]-q[13]*q[10])+q[17]-(q[13]+q[10]-q[13]*q[10])*q[17])*q[42]))
            w[10] = q[5]
            w[11] = q[5]*(1-q[7])
            w[12] = q[5]
            w[13] = (q[6]+q[16]*q[41]-q[6]*q[16]*q[41])*(1-(q[10]+q[14]-q[10]*q[14]))
            w[14] = (q[9]+q[7]-q[9]*q[7])
            w[15] = (q[54]+q[99]-q[54]*q[99])
            w[16] = ((q[4]+q[46]-q[4]*q[46])+q[78]-(q[4]+q[46]-q[4]*q[46])*q[78])
            w[17] = q[11]
            w[18] = (q[11]+q[60]-q[11]*q[60])
            w[19] = q[12]
            w[20] = (q[9]+q[9]*q[48]-q[9]*q[9]*q[48])
            w[21] = ((q[7]+q[8]*q[9]-q[7]*q[8]*q[9])+q[48]*q[9]-(q[7]+q[8]*q[9]-q[7]*q[8]*q[9])*q[48]*q[9])
            w[22] = q[35]
            w[23] = q[22]
            w[24] = q[23]
            w[25] = (q[24]+q[41]-q[24]*q[41])
            w[26] = ((1-q[29])+(1-q[28])-(1-q[29])*(1-q[28]))
            w[27] = ((((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])+q[13]-((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])*q[13])+q[42]*q[48]-(((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])+q[13]-((q[10]+q[11]-q[10]*q[11])+q[12]-(q[10]+q[11]-q[10]*q[11])*q[12])*q[13])*q[42]*q[48])
            w[28] = q[78]*(1-q[0])
            w[29] = q[78]*(1-q[0])
            w[30] = p[1]
            w[31] = p[2]
            w[32] = p[3]
            w[33] = p[3]
            w[34] = p[4]
            w[35] = ((((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])+q[33]-((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])*q[33])+q[34]-(((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])+q[33]-((q[30]+q[3]-q[30]*q[3])+q[32]-(q[30]+q[3]-q[30]*q[3])*q[32])*q[33])*q[34])
            w[36] = q[35]
            w[37] = q[36]
            w[38] = q[37]
            w[39] = (q[38]+q[50]-q[38]*q[50])
            w[40] = q[39]
            w[41] = q[40]
            w[42] = q[25]*(1-q[26])
            w[43] = q[39]
            w[44] = (q[43]+q[78]-q[43]*q[78])
            w[45] = q[44]
            w[46] = q[49]
            w[47] = q[16]
            w[48] = ((q[45]+q[47]-q[45]*q[47])+q[25]*(1-q[26])-(q[45]+q[47]-q[45]*q[47])*q[25]*(1-q[26]))
            w[49] = q[39]
            w[50] = (q[52]+q[97]-q[52]*q[97])
            w[51] = q[3]
            w[52] = (q[31]+q[51]-q[31]*q[51])
            w[53] = ((((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])+q[37]*q[33]-((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])*q[37]*q[33])+q[37]*q[34]-(((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])+q[37]*q[33]-((q[52]+q[97]*q[96]-q[52]*q[97]*q[96])+q[37]*q[32]-(q[52]+q[97]*q[96]-q[52]*q[97]*q[96])*q[37]*q[32])*q[37]*q[33])*q[37]*q[34])
            w[54] = q[53]*q[52]
            w[55] = q[53]*q[37]
            w[56] = (q[55]+q[99]-q[55]*q[99])
            w[57] = (q[9]+q[48]-q[9]*q[48])
            w[58] = q[9]*q[48]
            w[59] = q[9]*q[48]
            w[60] = (q[15]+q[56]-q[15]*q[56])
            w[61] = (q[73]*q[72]+q[70]-q[73]*q[72]*q[70])
            w[62] = q[61]
            w[63] = (q[61]*q[62]+q[51]-q[61]*q[62]*q[51])
            w[64] = q[63]
            w[65] = q[64]
            w[66] = q[4]
            w[67] = ((((q[66]+q[104]-q[66]*q[104])+q[74]-(q[66]+q[104]-q[66]*q[104])*q[74])+q[77]-((q[66]+q[104]-q[66]*q[104])+q[74]-(q[66]+q[104]-q[66]*q[104])*q[74])*q[77])+q[83]*q[84]-(((q[66]+q[104]-q[66]*q[104])+q[74]-(q[66]+q[104]-q[66]*q[104])*q[74])+q[77]-((q[66]+q[104]-q[66]*q[104])+q[74]-(q[66]+q[104]-q[66]*q[104])*q[74])*q[77])*q[83]*q[84])
            w[68] = q[67]
            w[69] = q[67]
            w[70] = q[69]
            w[71] = q[69]
            w[72] = q[70]
            w[73] = q[71]
            w[74] = q[79]
            w[75] = q[85]
            w[76] = q[85]
            w[77] = (q[76]+q[75]-q[76]*q[75])
            w[78] = (q[67]+q[35]-q[67]*q[35])
            w[79] = q[67]
            w[80] = q[78]
            w[81] = q[79]
            w[82] = p[18]
            w[83] = q[82]
            w[84] = q[82]
            w[85] = p[20]
            w[86] = (q[80]+q[9]-q[80]*q[9])
            w[87] = q[86]*(q[80]+q[9]-q[80]*q[9])*(1-q[90])
            w[88] = (q[86]*q[90]+q[86]*p[12]-q[86]*q[90]*q[86]*p[12])
            w[89] = ((q[80]*q[93]+p[12]-q[80]*q[93]*p[12])+p[15]-(q[80]*q[93]+p[12]-q[80]*q[93]*p[12])*p[15])
            w[90] = (q[89]+(q[74]+q[80]-q[74]*q[80])*q[93]-q[89]*(q[74]+q[80]-q[74]*q[80])*q[93])*(1-q[87])
            w[91] = (q[87]+q[94]-q[87]*q[94])*p[25]
            w[92] = q[90]*p[32]
            w[93] = q[91]*(1-q[92])
            w[94] = (1-p[35])*q[80]
            w[95] = (p[27]+p[28]-p[27]*p[28])
            w[96] = q[95]
            w[97] = (q[96]+q[100]-q[96]*q[100])
            w[98] = q[50]*q[97]
            w[99] = q[53]*q[97]
            w[100] = (p[29]+q[102]-p[29]*q[102])
            w[101] = q[97]
            w[102] = q[9]
            w[103] = (q[103]+q[105]-q[103]*q[105])
            w[104] = q[9]
            w[105] = q[9]
            w[106] = q[9]
            w[107] = q[9]
            w[108] = q[9]
            w[109] = ((q[64]+p[7]-q[64]*p[7]) / (1 + np.exp((t - p[42]))))*(1-q[122])
            w[110] = ((q[65]+p[8]-q[65]*p[8]) / (1 + np.exp((t - p[41]))))*(1-q[122])
            w[111] = q[127]*(1-p[39])
            w[112] = q[113]*(1-q[122])
            w[113] = q[118]*q[111]*(1-q[124])
            w[114] = q[113]
            w[115] = (q[112]+q[120]-q[112]*q[120])
            w[116] = q[109]*q[128]*(1-q[122])*(1-p[39])
            w[117] = q[112]*(1-q[122])
            w[118] = q[136]*(1-p[40])
            w[119] = q[137]*(1-p[39])
            w[120] = (q[110]*(1-q[122])+q[129]-q[110]*(1-q[122])*q[129])
            w[121] = q[113]*q[116]
            w[122] = (q[121]+q[142]*q[144]-q[121]*q[142]*q[144])
            w[123] = q[120]
            w[124] = q[118]*(1-q[120])
            w[125] = q[131]*q[133]
            w[126] = q[110]
            w[127] = (q[130]*q[126]*q[125]+q[112]*q[125]-q[130]*q[126]*q[125]*q[112]*q[125])
            w[128] = q[109]*(1-q[122])
            w[129] = ((q[112]+q[110]-q[112]*q[110])+q[134]-(q[112]+q[110]-q[112]*q[110])*q[134])
            w[130] = q[116]
            w[131] = (q[116]+q[112]-q[116]*q[112])
            w[132] = (q[116]+q[112]-q[116]*q[112])
            w[133] = (q[132]+q[131]-q[132]*q[131])
            w[134] = q[133]
            w[135] = q[133]*q[131]
            w[136] = q[135]
            w[137] = q[125]
            w[138] = (p[24]+q[59]-p[24]*q[59])*(p[23]+q[58]-p[23]*q[58])*(p[17]+q[21]-p[17]*q[21])*q[114]*(p[5]+p[6]-p[5]*p[6])*q[151]*q[119]*q[118]*q[111]*p[34]*q[157]*(1-q[141])*(1-q[143])*(1-q[140])
            w[139] = q[138]*q[111]*q[118]*(1-q[140])
            w[140] = (p[24]+q[59]-p[24]*q[59])*(p[12]+p[13]-p[12]*p[13])*q[114]*q[152]*q[117]*q[118]*(1-q[138])*(1-q[144])*(1-q[139])*(1-q[148])
            w[141] = q[140]*(1-q[138])*(1-q[139])
            w[142] = ((p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[118]*q[117]*q[111]*q[114]+p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[143]*q[121]-p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[118]*q[117]*q[111]*q[114]*p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[143]*q[121])+p[14]*q[144]-(p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[118]*q[117]*q[111]*q[114]+p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[143]*q[121]-p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[118]*q[117]*q[111]*q[114]*p[14]*((p[16]+p[15]-p[16]*p[15])+q[27]-(p[16]+p[15]-p[16]*p[15])*q[27])*q[143]*q[121])*p[14]*q[144])*(1-q[139])*(1-q[160])*(1-(p[21]+q[57]-p[21]*q[57]))
            w[143] = p[14]*q[142]
            w[144] = q[142]
            w[145] = (((p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[111]*q[151]*p[34]+p[22]*p[14]*q[111]*q[151]*q[157]-(p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[111]*q[151]*p[34]*p[22]*p[14]*q[111]*q[151]*q[157])+q[160]-((p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[111]*q[151]*p[34]+p[22]*p[14]*q[111]*q[151]*q[157]-(p[21]+q[57]-p[21]*q[57])*p[22]*p[14]*q[111]*q[151]*p[34]*p[22]*p[14]*q[111]*q[151]*q[157])*q[160])*(1-q[138])*(1-q[142])*(1-q[140])
            w[146] = (p[22]*q[145]+(p[21]+q[57]-p[21]*q[57])*q[148]-p[22]*q[145]*(p[21]+q[57]-p[21]*q[57])*q[148])*(1-q[139])*(1-q[141])*(1-q[143])
            w[147] = q[145]
            w[148] = (p[21]+q[57]-p[21]*q[57])*p[22]*q[111]*q[151]*(1-q[145])*(1-q[138])*(1-q[140])
            w[149] = q[148]
            w[150] = q[148]
            w[151] = (q[115]*q[120]+q[115]*q[157]-q[115]*q[120]*q[115]*q[157])*(1-q[154])*(1-p[38])
            w[152] = (q[115]*q[154]+q[115]*(p[12]+p[13]-p[12]*p[13])-q[115]*q[154]*q[115]*(p[12]+p[13]-p[12]*p[13]))
            w[153] = q[120]*q[159]
            w[154] = (((((q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))+q[120]*q[159]*(1-q[151])-(q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))*q[120]*q[159]*(1-q[151]))+q[142]-((q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))+q[120]*q[159]*(1-q[151])-(q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))*q[120]*q[159]*(1-q[151]))*q[142])+q[148]-(((q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))+q[120]*q[159]*(1-q[151])-(q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))*q[120]*q[159]*(1-q[151]))+q[142]-((q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))+q[120]*q[159]*(1-q[151])-(q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))*q[120]*q[159]*(1-q[151]))*q[142])*q[148])+p[37]-((((q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))+q[120]*q[159]*(1-q[151])-(q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))*q[120]*q[159]*(1-q[151]))+q[142]-((q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))+q[120]*q[159]*(1-q[151])-(q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))*q[120]*q[159]*(1-q[151]))*q[142])+q[148]-(((q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))+q[120]*q[159]*(1-q[151])-(q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))*q[120]*q[159]*(1-q[151]))+q[142]-((q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))+q[120]*q[159]*(1-q[151])-(q[153]*(1-q[151])+q[136]*q[159]*(1-q[151])-q[153]*(1-q[151])*q[136]*q[159]*(1-q[151]))*q[120]*q[159]*(1-q[151]))*q[142])*q[148])*p[37])
            w[155] = (q[151]*p[26]+q[160]*p[26]-q[151]*p[26]*q[160]*p[26])*(1-q[159])*(1-q[148])
            w[156] = p[31]
            w[157] = q[156]
            w[158] = q[154]*p[33]
            w[159] = q[155]*(1-q[158])
            w[160] = (1-p[36])*q[120]
            dqdt = act_f(af, x=w, b=p[0], u=threshold_values) - decay_rates*q
            return dqdt

        return system_ode

    def solve(self, t_span=(0, 50), t_eval=None, grinitial_conditions=None, grmethod=None, grout_of_the_network_arguments_eval=None, af=None, decay_rates_list=None, grthreshold_values=None):
        if grinitial_conditions is None:
            grinitial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
        if grmethod is None:
            grmethod = 'LSODA'
        if grout_of_the_network_arguments_eval is None:
            grout_of_the_network_arguments_eval = np.array([10, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 0.0, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0.0, 15, 15])
        if decay_rates_list is None:
            decay_rates_list = np.full(self.n, 1)
        if af is None:
            af = 'sigmoid'
        if grthreshold_values is None:
            grthreshold_values = np.full(self.n, 0.5)

        ode_system = self.create_ode_system(af, decay_rates_list, grthreshold_values)
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)

        if self.out_of_network_arguments is None:
            sol = solve_ivp(
                ode_system,
                t_span,
                grinitial_conditions,
                method=grmethod,
                t_eval=t_eval,
                dense_output=True
            )
        else:
            sol = solve_ivp(
                ode_system,
                t_span,
                grinitial_conditions,
                args=grout_of_the_network_arguments_eval,
                method=grmethod,
                t_eval=t_eval,
                dense_output=True
            )

            resultados = {
                'tiempos': sol.t,
                'genes': {}
            }
            for i, gen in enumerate(self.array_of_elements):
                resultados['genes'][gen] = {
                    'valores': sol.y[i],
                    'inicial': sol.y[i, 0],
                    'final': sol.y[i, -1],
                    'max': np.max(sol.y[i]),
                    'min': np.min(sol.y[i]),
                    'mean': np.mean(sol.y[i])
                }

        return resultados

"""If someone wants to transform from boolean logic to Fuzzy logic"""
#Tranformation to get new
import re
import string
import os
import shutil
from pathlib import Path
import tempfile


class Transformationtrial:
    def __init__(self):
        self.ref_q = {}
        self.ref_p = {}
        # Crear directorio persistente para logs (opcional)
        self.log_dir = "./logs_conversion"
        os.makedirs(self.log_dir, exist_ok=True)
    def procesar_archivo_simple(self, file, file_nodes, file_inputs):
        """
        Versión más directa sin función auxiliar create_temporary_file
        """
        if file is None or file_nodes is None or file_inputs is None:
            return None, None, None, "No hay archivo"

        try:
            # 1. CREAR ARCHIVOS TEMPORALES DE SALIDA DIRECTAMENTE
            fd_boolean, ruta_boolean = tempfile.mkstemp(suffix='.txt', prefix='boolean_')
            fd_nodes, ruta_nodes = tempfile.mkstemp(suffix='.txt', prefix='nodes_')
            fd_inputs, ruta_inputs = tempfile.mkstemp(suffix='.txt', prefix='inputs_')

            # 2. PROCESAR BOOLEANO (usando archivo original directamente)
            with open(file.name, 'r', encoding='utf-8') as f_in, \
                 os.fdopen(fd_boolean, 'w') as f_out:
                for linea in f_in:
                    linea_limpia = linea.strip()
                    if linea_limpia:
                        f_out.write(self.look_for_parentheses(linea_limpia) + '\n')

            # 3. PROCESAR NODOS
            with open(file_nodes.name, 'r', encoding='utf-8') as f_in, \
                 os.fdopen(fd_nodes, 'w') as f_out:
                lineas = [l.strip() for l in f_in if l.strip()]
                self.ref_q = {f'q[{i}]': linea for i, linea in enumerate(lineas)}
                for element, psub in self.ref_q.items():
                    f_out.write(f"{psub} --> {element}\n")

            # 4. PROCESAR INPUTS
            with open(file_inputs.name, 'r', encoding='utf-8') as f_in, \
                 os.fdopen(fd_inputs, 'w') as f_out:
                lineas = [l.strip() for l in f_in if l.strip()]
                self.ref_p = {f'p[{i}]': linea for i, linea in enumerate(lineas)}
                for element, psub in self.ref_p.items():
                    f_out.write(f"{psub} --> {element}\n")

            return ruta_boolean, ruta_nodes, ruta_inputs,'Success'

        except Exception as e:
            return None, None, None, f"Error: {str(e)}"
    def evaluate_boolean(self,left_char, booref, right_char) -> str:
        if booref == '|':
            new_expression = rf'({left_char}+{right_char}-{left_char}*{right_char})'
            return new_expression
        elif booref == '&':
            new_expression = rf'{left_char}*{right_char}'
            return new_expression
        else:
            return 'Unknownsymbol'
    #this handles the ! case
    def evaluate_negative(self,right_char):
        return rf'(1-{right_char})'
    #This function handles patterns without parenthesis a|(b!c)
    def all_characters(self,expression):
        patternneg = r'!(\(.+\)|\w+|\([^)]+\)|)'
        matchnneg = re.search(patternneg,expression)
        if matchnneg is not None:
            nptt = self.evaluate_negative(matchnneg.group(1))
            expression = expression.replace(matchnneg.group(0),nptt)
            return self.all_characters(expression)
        else:
            #Iniciamos con los patrones and
            patterand = r'([^&|]+)(&)([^&|]+)'
            decomposed_and = re.search(patterand,expression)
            #Si existe el patron and entonces continuamos
            if decomposed_and is not None:
                f_match = decomposed_and.group(0)  # Expresión completa
                l_match = decomposed_and.group(1)  # Parte izquierda
                op_match = decomposed_and.group(2)  # Operador (& o |)
                r_match = decomposed_and.group(3)  # Parte derecha
                expr_to_replace = self.evaluate_boolean(l_match, op_match, r_match)
                # Reemplazar y continuar recursión
                new_expression = expression.replace(f_match, expr_to_replace)
                #recursión necesaria para verificar que no hay más patrones and
                return self.all_characters(new_expression)
            else:
                #Cuando se agotaron los patrones and iniciamos este loop |
                pattern = r'([^&|]+)(\|)([^&|]+)'
                decomposed_expressions = re.search(pattern, expression)
                #mismo patrón para eliminar parentesis
                if decomposed_expressions is not None:
                    f_match = decomposed_expressions.group(0)  # Expresión completa
                    l_match = decomposed_expressions.group(1)  # Parte izquierda
                    op_match = decomposed_expressions.group(2)  # Operador (& o |)
                    r_match = decomposed_expressions.group(3)  # Parte derecha
                    expr_to_replace = self.evaluate_boolean(l_match, op_match, r_match)
                    # Reemplazar y continuar recursión
                    new_expression = expression.replace(f_match, expr_to_replace)
                    return self.all_characters(new_expression)
                else:
                    return expression
    #iteration function to ressolve parentheses first
    def look_for_parentheses(self,expression):
        expression=expression.replace(' ','')
        if expression.count('(') == 0 and expression.count(')') == 0:
            expression=self.all_characters(expression)
            return expression
        expression_keys = {}
        i = 0
        while '(' in expression:
            # Buscar el par de paréntesis más interno
            match = re.search(r'\(([^()]+)\)', expression)
            if match:
                contenido = match.group(1)
                print("Se está agregando una clave al diccionario")
                expression_keys[match.group(0)]=self.marker(i)
                expression = expression.replace(match.group(0),expression_keys[match.group(0)])
                i+=1
            else:
                break
        expression=self.all_characters(expression)
        print(f"Expression before replacing with the dictionary :{expression}")
        if expression_keys is not None:
            dict_invertido = {k: expression_keys[k] for k in reversed(list(expression_keys.keys()))}
            for i,j in dict_invertido.items():
                print(f"La clave es {i} y el valor es {j}")
                print(f"Al resolver los parentesis")
                expression=expression.replace(j,i)
                insidepar=self.all_characters(i.strip(')('))
                expression= expression.replace(i,insidepar)
        return expression
    def marker(self,marker_number :int=0) -> str:
        return f"_A{marker_number}_"

from dataclasses import dataclass
import matplotlib.pyplot as plt
import math
import gradio as gr
import random
import plotly.graph_objects as go
class NetworkInterface(NetworkEquations):
    def __init__(self,features :dict):
        super().__init__()
        self.value_sliders = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
        self.max_sliders_out = [100] + (len(self.out_of_network_arguments)-3)*[1]+2*[100]
        self.value_sliders_out = np.array([10, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 0.0, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0, 0.0, 15, 15])
        self.distribution_type = None
        self.features = features
        self.subnetwork_options = list(features.keys())
        self.last_results = None # Initialize last_results
        self.gradio_interface()
    """ mathplotlib was changed to pyplot by doing this we are improving the performance, instead of generating
    the 165 plot we just store the data and plot it as the user requests it"""

    def plot_results(self, resultados :dict=None,options: list=None) -> go.Figure:
        if options is None:
            options = ['OXPHOS','IL10','TGFB','RORGT','IL21','IL17','BCL6']
        if len(options) > 16:
            fig = go.Figure()
            return go.Figure().update_layout(title=f"Error: Change one of the nodes to add it to your graphic")
        else:
            fig = go.Figure()
            t = resultados['tiempos']
            for option in options:
                fig.add_trace(go.Line(x=t,y=resultados['genes'][option]['valores'],mode='lines',name=option,line=dict(width=2),
                                      hovertemplate=f'{option}<br>Time: %{{x:.2f}}<br>Value: %{{y:.4f}}'))
            Titulo={
            'text': "Nodes expression",
            'x': 0.5,           #  Centrado horizontal (0=izquierda, 0.5=centro, 1=derecha)
            'y': 0.95,          #  Posición vertical (0=abajo, 1=arriba)
            'xanchor': 'center', #  Anclaje horizontal
            'yanchor': 'top'     #
            }

            fig.update_layout(
                title=Titulo,
                yaxis=dict(range=[-0.1,1], autorange=False),
                xaxis_title='Time',
                yaxis_title='Expression',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
        return fig
    def design_and_personalize_subplots(self, resultados):
        n_co = 5
        n_rows = math.ceil(len(self.features.keys())/n_co)
        fig2, axis = plt.subplots(n_rows, n_co, figsize=(16, 4.3*n_rows))
        axis = axis.flatten()
        categorias = list(self.features.keys())[:19]
        for idx, (axi, subr) in enumerate(zip(axis, categorias)):
            if subr == 'Phenotypes':
                labels = self.features[subr]['labels']
                axi.set_xticks(range(0, len(labels)))
                axi.set_xticklabels(labels, rotation=90, ha='right', fontsize=8)
                axi.set_ylim(0,1)
                phenotype_population = [resultados['genes'][k]['mean'] for k in self.features[subr]['variables']  if k in resultados['genes']]
                axi.bar(labels,phenotype_population, color='purple')
                axi.legend(bbox_to_anchor=(0., 1.3, 1., .09), loc=1, ncol=1, mode="expand",borderaxespad=0., prop={'size': 8}, title='Th phenotypes', fontsize=8)
            else:
                for inx, (gen, col, sty, label) in enumerate(zip(self.features[subr]['variables'],
                                                self.features[subr]['colores'],
                                                self.features[subr]['estilos'],
                                                self.features[subr]['labels'])):
                    data = resultados['genes'][gen]
                    axi.plot(resultados['tiempos'], data['valores'], label=label,
                              color=col, linestyle=sty, linewidth=2)
                    axi.legend(bbox_to_anchor=(0., 1.5, 1., .02), loc=1, ncol=1, mode="expand", borderaxespad=0., prop={'size': 8},title=subr,fontsize=8)
            axi.set_xlabel('Time',fontsize=8)
            axi.set_ylabel('Expression',fontsize=8)
            axi.set_ylim(-0.1,1)
            axi.grid(True, alpha=0.3)
        for j in range(idx + 1, len(axis)):
            axis[j].set_visible(False)
        plt.show()
        plt.tight_layout(h_pad=1.80)
        return fig2
    def create_Plot(self):
        if self.last_results is None:
            return go.Figure().update_layout(title='Run a network simulation first')
        else:
            Figura = self.design_and_personalize_subplots(self.last_results)
            return Figura
    def create_selected_subnetwork_plot(self, resultados: dict, subnetwork_name: str) -> go.Figure:
        if subnetwork_name not in self.features:
            return go.Figure().update_layout(title=f"Subred '{subnetwork_name}' no encontrada")
        config = self.features[subnetwork_name]
        fig = go.Figure()
        t = resultados['tiempos']
        for i, (var, label) in enumerate(zip(self.features[subnetwork_name]['variables'], self.features[subnetwork_name]['labels'])):
            if var in resultados['genes']:
                fig.add_trace(go.Line(
                    x=t,
                    y=resultados['genes'][var]['valores'],
                    mode='lines',
                    name=label,
                    line=dict(width=2),
                    hovertemplate=f'{label}<br>Time: %{{x:.2f}}<br>Value: %{{y:.4f}}'
                ))
        Titulo={
        'text': f"{subnetwork_name}",
        'x': 0.5,           #
        'y': 0.95,          # )
        'xanchor': 'center', #
        'yanchor': 'top'     #
            }
        fig.update_layout(
            title=Titulo,
            xaxis_title="Time",
            yaxis_title="Expression",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )

        return fig

    def gradio_interface(self, n_columns: int = 5):
        def actualizar_titulo(a_function, method):
            return f"""
            <div style="background-color:#4c8aad; padding:10px; border-radius:5px;">
                <h3 style="margin:0; text-align: center;">Immune network simulator</h3>
                <p style="margin:5px 0; text-align: center;">
                    Network: {self.n} Nodes |
                    Activation: {a_function} |
                    Method: {method}
                </p>
            </div>
            """

        def actualizar_parametros_fuera_de_red(a_function, method, num_params):
            return f"""
            <div style="background-color:#4c8aad; padding:10px; border-radius:5px;">
                <h3 style="margin:0; text-align: center;"> Inputs to simulate this immune network</h3>
                <p style="margin:5px 0; text-align: center;">
                    Network: {self.n} Nodes |
                    Activation: {a_function} |
                    Method: {method} |
                    Number of inputs: {num_params}
                </p>
            </div>
            """
        def update_decay_rates(a_function, method, num_params):
            return f"""
            <div style="background-color:#4c8aad; padding:10px; border-radius:5px;">
                <h3 style="margin:0; text-align: center;"> Decay rates to simulate this immune network</h3>
                <p style="margin:5px 0; text-align: center;">
                    Network: {self.n} Nodes |
                    Activation: {a_function} |
                    Method: {method} |
                    Number of decay rates: {num_params}
                </p>
            </div>
            """
        def update_umbral_values(a_function, method, num_params):
            return f"""
            <div style="background-color:#4c8aad; padding:10px; border-radius:5px;">
                <h3 style="margin:0; text-align: center;">Thresholds to simulate this immune network</h3>
                <p style="margin:5px 0; text-align: center;">
                    Network: {self.n} Nodes |
                    Activation: {a_function} |
                    Method: {method} |
                    Number of threshold values: {self.n}
                </p>
            </div>
            """

        def actualizar_interfaz(act, meth):
            return actualizar_titulo(act, meth), actualizar_parametros_fuera_de_red(act, meth, len(self.out_of_network_arguments)), update_decay_rates(act, meth, self.n), update_umbral_values(act, meth, self.n)
        with gr.Blocks(title="Network simulator") as demo:

            with gr.Row(variant="compact"):
                gr.Markdown("Solving ODE rapidly")
                gr.Markdown("Exploring networks more efficiently")
            with gr.Row(variant="compact"):
                        activation = gr.Dropdown(
                                choices=['sigmoid', 'tanh', 'relu'],
                                label="Activation function",
                                value='sigmoid'
                            )
                        method = gr.Dropdown(
                            choices=['TRBDF2', 'DOP853', 'RadauIIA5', 'Rodas5', 'LSODA'],
                            label="ODE Solver",
                            value='LSODA'
                        )
            with gr.Row(variant="compact"):
                distribution = gr.Dropdown(choices=['normal','uniform','laplace'],value='uniform',label='Distribution to simulate network dynamics')

            with gr.Row(variant="compact"):
                        solve_btn = gr.Button("Simulate network dynamics", variant="primary")
            a_function = 'sigmoid'
            n_columns = 5
            with gr.Tabs():
                with gr.Tab("Initial conditions"):
                    output = gr.HTML(actualizar_titulo(a_function, method))
                    with gr.Row():
                        with gr.Column():
                            restart_initial_conditions = gr.Button('Reset initial conditions',variant='secondary')
                        with gr.Column():
                            set_random_extremal_values = gr.Button('Set extreme values',variant='secondary')
                    todos_sliders_ic = []
                    sliders_reference = dict(zip(self.array_of_elements,self.value_sliders))
                    hide_sliders = ['IL10OUT','IL6OUT','IL33OUT','IL18OUT','IL12OUT']
                    n_filas_ic = (self.n + n_columns - 1) // n_columns
                    for j in range(n_filas_ic):
                        with gr.Row(variant="compact"):
                            inicio = j * 5
                            fin = min((j + 1) * 5, self.n)
                            fila_sliders = []
                            for idx in range(inicio, fin):
                                gen = self.array_of_elements[idx]
                                val = self.value_sliders[idx]
                                slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    label=f'{gen}',
                                    value=val,
                                    interactive=True,
                                    visible=gen not in hide_sliders
                                )
                                fila_sliders.append(slider)
                                todos_sliders_ic.append(slider)
                    distribution.change(
                        fn=self.set_distribution,
                        inputs=distribution,
                        outputs=todos_sliders_ic
                    )
                    restart_initial_conditions.click(fn=self.restart_sliders_value,outputs=todos_sliders_ic)
                    set_random_extremal_values.click(fn=self.set_extremal_distribution,outputs=todos_sliders_ic)
                with gr.Tab("Inputs"):
                    output2 = gr.HTML(actualizar_parametros_fuera_de_red(a_function, method, len(self.out_of_network_arguments)))
                    all_sliders_out = []
                    n_filas_out = (len(self.out_of_network_arguments) + n_columns - 1) // n_columns
                    with gr.Row():
                        with gr.Column():
                            reset_network_params = gr.Button('Reset input values', variant='secondary')
                        with gr.Column():
                            set_extreme_network_params = gr.Button('Random values for input values', variant='secondary')
                    with gr.Column():
                        for j in range(n_filas_out):
                            with gr.Row(variant="compact"):
                                inicio = j * 5
                                fin = min((j + 1) * 5, len(self.out_of_network_arguments))
                                fila_sliders_out = []
                                for idx in range(inicio, fin):
                                    out = self.out_of_network_arguments[idx]
                                    slider = gr.Slider(
                                        minimum=0.0,
                                        maximum=self.max_sliders_out[idx],
                                        step=0.01,
                                        label=f'{self.out_of_network_arguments[idx]}',
                                        value=self.value_sliders_out[idx],
                                        interactive=True
                                    )
                                    fila_sliders_out.append(slider)
                                    all_sliders_out.append(slider)
                        with gr.Row():
                            outputalpha = gr.HTML(update_decay_rates(a_function, method, len(self.out_of_network_arguments)))
                        with gr.Row():
                            with gr.Column():
                                reset_decay_rates_sliders = gr.Button('Reset decay rates', variant='secondary')
                            with gr.Column():
                                random_decay_rates_sliders = gr.Button('Set random decay rates', variant='secondary')
                            with gr.Column():
                                extreme_decay_rates_sliders = gr.Button('Set extreme decay rates', variant='secondary')
                        todos_sliders_alpha = []
                        fila_sliders_alpha = []
                        n_filas_alpha= (self.n + n_columns - 1) // n_columns
                        for s in range(n_filas_alpha):
                            with gr.Row(variant="compact"):
                                start = s * n_columns
                                end = min((s + 1) * n_columns, self.n)
                                value_decay_rate = [1 for j in range(self.n)]
                                for idx in range(start, end):
                                    val = (value_decay_rate[idx])
                                    par = self.array_of_elements[idx]
                                    slider = gr.Slider(
                                        minimum=0.0,
                                        maximum=2.0,
                                        step=0.01,
                                        label=f" Decay rate:({par})",
                                        value=val,
                                        interactive=True
                                    )
                                    fila_sliders_alpha.append(slider)
                                    todos_sliders_alpha.append(slider)
                with gr.Tab("Threshold values"):
                    outputumbral = gr.HTML(update_umbral_values(a_function, method, self.n))
                    with gr.Row():
                        with gr.Column():
                            reset_threshold_values = gr.Button('Reset threshold values', variant='secondary')
                        with gr.Column():
                            set_extreme_threshold_values = gr.Button('Set extreme threshold values', variant='secondary')
                    with gr.Column():
                        todos_sliders_umbral = []
                        fila_sliders_umbral = []
                        n_filas_alpha = (self.n + n_columns - 1) // n_columns
                        for s in range(n_filas_alpha):
                            with gr.Row(variant="compact"):
                                start = s * n_columns
                                end = min((s + 1) * n_columns, self.n)
                                value_decay_rate = [1 for j in range(self.n)]
                                for idx in range(start, end):
                                    val = (value_decay_rate[idx])
                                    par = self.array_of_elements[idx]
                                    slider = gr.Slider(
                                        minimum=0.0,
                                        maximum=1.0,
                                        step=0.01,
                                        label=f" Threshold: U({par})",
                                        value=0.5,
                                        interactive=True
                                    )
                                    fila_sliders_umbral.append(slider)
                                    todos_sliders_umbral.append(slider)
                    with gr.Row():
                        with gr.Column():
                            select_box = gr.CheckboxGroup(
                                    choices=list(self.array_of_elements),
                                    value=['OXPHOS','IL10','TGFB','RORGT','IL21','IL17','BCL6'],
                                    label="Choose the nodes to plot (15 nodes)")
                        with gr.Column():
                            figura = gr.Plot()
                            select_box.change(
                                fn=self.update_general_plot,
                                inputs=select_box,
                                outputs=figura)

                with gr.Tab("Plot customization"):
                    with gr.Row():
                        with gr.Column():
                            subnetwork_selector = gr.Dropdown(
                                    choices=self.subnetwork_options,
                                    value=self.subnetwork_options[0] if self.subnetwork_options else "",
                                    label="Choose Subnetwork to display",
                                    interactive=True
                                )
                            microplots = gr.Plot()
                with gr.Tab("Results"):
                    gr.Markdown("Summary of the main results of cellular dynamics")
                    with gr.Column():
                        GButton = gr.Button("Plot main cellular dynamics", variant="primary")
                        GPlot = gr.Plot()
                with gr.Tab("Convert and process your own boolean networks"):
                    with gr.Row():
                        procesar_btn = gr.Button("Process file", variant="primary", size="lg")
                        limpiar_btn = gr.Button("Delete file", variant="secondary")
                    with gr.Row():
                        with gr.Column():
                            archivo_boolean = gr.File(
                                label="📄 Boolean rules file",
                                file_types=[".txt"]
                            )
                        with gr.Column():
                            archivo_nodes = gr.File(
                                label="📄 Nodes file",
                                file_types=[".txt"]
                            )
                        with gr.Column():
                            archivo_inputs = gr.File(
                                label="📄 Inputs file",
                                file_types=[".txt"]
                            )
                    with gr.Row():
                        with gr.Column():
                            archivo_out_boolean = gr.File(
                                label="📥 Probablistic expression file",
                                interactive=False
                            )
                        with gr.Column():
                            archivo_out_nodes = gr.File(
                                label="📥 Nodes reference file",
                                interactive=False
                            )
                        with gr.Column():
                            archivo_out_inputs = gr.File(
                                label="📥 Inputs reference file",
                                interactive=False
                            )

                    info_text = gr.Textbox(label="ℹ️ Information", lines=3)
                with gr.Tab("General information"):
                    gr.Markdown("Information about the network proposal")
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Componente para subir imagen
                            network_image = gr.Image(
                                label="thumbnail of the paper ",
                                type="filepath",  # Guarda la ruta del archivo
                                height=300
                            )
                        with gr.Column(scale=1):
                            # Caja de texto para descripción
                            image_description = gr.Textbox(
                                label="This network represents the network's proposal of the article",
                                placeholder=".",
                                lines=8
                            )

            variables_to_control = todos_sliders_ic+all_sliders_out+todos_sliders_alpha+todos_sliders_umbral+[activation]+[method]+[select_box]
            activation.change(
                        fn=actualizar_interfaz,
                        inputs=[activation, method],
                        outputs=[output, output2, outputalpha,outputumbral]
                    )
            method.change(
                fn=actualizar_interfaz,
                inputs=[activation, method],
                outputs=[output, output2, outputalpha, outputumbral]
            )
            solve_btn.click(
                fn=self.after_push_button,
                inputs=variables_to_control,
                outputs=[figura,microplots]
            )
            subnetwork_selector.change(
                fn=self.update_selected_subnetwork,
                inputs=subnetwork_selector,
                outputs=microplots
            )
            GButton.click(
                fn = self.create_Plot,
                outputs=GPlot
            )

            reset_network_params.click(
                fn=self.reset_network_parameters,
                outputs=all_sliders_out
            )

            set_extreme_network_params.click(
                fn=self.set_extreme_network_parameters,
                outputs=all_sliders_out
            )
            reset_decay_rates_sliders.click(
                fn=self.set_decay_rates,
                outputs=todos_sliders_alpha
            )

            random_decay_rates_sliders.click(
                fn=self.set_random_decay_rates,
                outputs=todos_sliders_alpha
            )
            extreme_decay_rates_sliders.click(
                fn=self.set_extreme_decay_rates,
                outputs=todos_sliders_alpha
            )
            reset_threshold_values.click(
                fn=self.reset_threshold_values,
                outputs=todos_sliders_umbral
            )
            set_extreme_threshold_values.click(
                fn=self.set_extreme_threshold_values,
                outputs=todos_sliders_umbral
            )
             #instance creation
            convertidor= Transformationtrial()
            procesar_btn.click(
                fn=convertidor.procesar_archivo_simple,  # o procesar_archivo
                inputs=[archivo_boolean, archivo_nodes, archivo_inputs],
                outputs=[archivo_out_boolean, archivo_out_nodes, archivo_out_inputs, info_text]
            )


        return demo.launch()

    # New function to update the general plot based on checkbox selection
    def update_general_plot(self, selected_nodes: list) -> go.Figure:
        if not hasattr(self, 'last_results') or self.last_results is None:
            return go.Figure().update_layout(title="Please run a simulation first to plot selected nodes.")
        return self.plot_results(self.last_results, selected_nodes)

    def after_push_button(self, *values):
        genes = values[:self.n]
        out_params = values[self.n:self.n+len(self.out_of_network_arguments)]
        decay_rates = values[self.n+len(self.out_of_network_arguments):self.n+len(self.out_of_network_arguments)+self.n]
        threshold_values = values[self.n+len(self.out_of_network_arguments)+self.n:self.n+len(self.out_of_network_arguments)+self.n+self.n]
        activation = values[self.n+self.n+self.n+len(self.out_of_network_arguments)]
        metodo = values[self.n+self.n+self.n+len(self.out_of_network_arguments)+1]
        selection_box_plot = values[self.n+self.n+self.n+len(self.out_of_network_arguments)+2]

        Resultados = self.solve(
            t_span=(0, 50),
            t_eval=None,
            grinitial_conditions=genes,
            grmethod=metodo,
            grout_of_the_network_arguments_eval=out_params,
            af=activation,
            decay_rates_list=decay_rates,
            grthreshold_values=threshold_values
        )
        self.last_results = Resultados

        Fig1 = self.plot_results(Resultados, selection_box_plot) # Pass selected nodes for initial general plot
        Fig2 = self.update_selected_subnetwork(self.subnetwork_name) # Default subnetwork plot
        return Fig1, Fig2

    def update_selected_subnetwork(self, subnetwork_name=None) -> go.Figure:
        if subnetwork_name is None:
            subnetwork_name = random.choice(self.features)
        if subnetwork_name not in self.features:
            return go.Figure().update_layout(title=f"Subred '{subnetwork_name}' no encontrada")

        Fig2 = self.create_selected_subnetwork_plot(self.last_results, subnetwork_name)
        self.subnetwork_name = subnetwork_name
        return Fig2
    def set_distribution(self,option):
        if option == 'normal':
            return np.absolute(np.random.normal(0.5,0.15, self.n)).tolist()
        elif option == 'uniform':
            return np.absolute(np.random.uniform(0.5,0.15, self.n)).tolist() # Changed to uniform
        elif option == 'laplace':
            return np.absolute(np.random.laplace(0.5,0.15,self.n)).tolist()
    def set_extremal_distribution(self, *values):
        values = np.random.choice([0,1],size=self.n).tolist()
        return values
    def restart_sliders_value(self,*values):
        values = self.value_sliders
        return values.tolist()
    def reset_network_parameters(self, *values):
        """Resetea los parámetros de red a sus valores iniciales"""
        return self.value_sliders_out.tolist()
    def set_extreme_network_parameters(self, *values):
        self.max_sliders_out
        extreme_values=[]
        for elemento in self.max_sliders_out:
            interval = np.linspace(0.0,elemento,5)
            valor = float(np.random.choice(interval))
            extreme_values.append(valor)
        return extreme_values
    def reset_threshold_values(self, *values):
        lista = np.full(self.n,0.5)
        return lista.tolist()
    def set_extreme_threshold_values(self, *values):
        """Establece valores extremos para los umbrales"""
        return np.random.choice([0.0, 1.0], size=self.n).tolist()
    def set_decay_rates(self,*values):
        arreglo=np.ones(self.n).tolist()
        return arreglo
    def set_random_decay_rates(self,*values):
        interval =np.linspace(0,1,10)
        random_array = np.random.choice(interval,size=self.n).tolist()
        return random_array
    def set_extreme_decay_rates(self,*values):
        extreme_array = np.random.choice([0,1], size=self.n).tolist()
        return extreme_array
if __name__ == "__main__":
    Subredes = {}

     # 1. METABOLISMO
    metabolismo_vars = ['MGlycolysis', 'MOXPHOS', 'MAMPATPratio', 'MAMPK']
    metabolismo_labels = ['Glycolysis', 'OXPHOS', 'AMP/ATP Ratio', 'AMPK']
    metabolismo_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    metabolismo_estilos = ['-', ':', '--', '-.']
    Subredes['Macrophages Metabolism'] = {'variables': metabolismo_vars, 'labels': metabolismo_labels,
                              'colores': metabolismo_colores, 'estilos': metabolismo_estilos}

    # 2. ACTIVIDAD DE MACRÓFAGOS
    macroactividad_vars = ['PHAGOCYTOSIS', 'PROCESSING', 'MHC2', 'PHAGOSOME']
    macroactividad_labels = ['Phagocytosis', 'Antigen Processing', 'MHC II', 'Endocytosis']
    macroactividad_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    macroactividad_estilos = ['-', ':', '--', '-.']
    Subredes['Macrophages Activity'] = {'variables': macroactividad_vars, 'labels': macroactividad_labels,
                                 'colores': macroactividad_colores, 'estilos': macroactividad_estilos}

    # 3. FACTORES DE TRANSCRIPCIÓN
    transcripcion_vars = ['MNFKB', 'MAP1', 'IRF3', 'IRF4', 'IRF7']
    transcripcion_labels = ['NF-κB', 'AP-1', 'IRF3', 'IRF4', 'IRF7']
    transcripcion_colores = ['lawngreen','blueviolet', 'tab:cyan', 'tab:red', 'black']
    transcripcion_estilos = ['-', ':', '--', '-.', '-']
    Subredes['Transcription Factors'] = {'variables': transcripcion_vars, 'labels': transcripcion_labels,
                                          'colores': transcripcion_colores, 'estilos': transcripcion_estilos}

    # 4. MACRÓFAGOS M1
    m1_vars = ['IFNABOUT', 'IL1BOUT', 'IL6OUT', 'IL12OUT', 'TNFAOUT']
    m1_labels = ['IFN-α', 'IL-1β', 'IL-6', 'IL-12', 'TNF-α']
    m1_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red', 'black']
    m1_estilos = ['-', ':', '--', '-.', '-']
    Subredes['M1 Markers'] = {'variables': m1_vars, 'labels': m1_labels,
                                'colores': m1_colores, 'estilos': m1_estilos}

    # 5. MACRÓFAGOS M2a
    m2a_vars = ['PPARG', 'IL10OUT', 'JMJD3', 'STAT6']
    m2a_labels = ['PPAR-γ', 'IL-10', 'JMJD3', 'STAT6']
    m2a_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    m2a_estilos = ['-', ':', '--', '-.']
    Subredes['M2a Markers'] = {'variables': m2a_vars, 'labels': m2a_labels,
                                 'colores': m2a_colores, 'estilos': m2a_estilos}

    # 6. MACRÓFAGOS M2b
    m2b_vars = ['ERK', 'IL10OUT']
    m2b_labels = ['ERK', 'IL-10']
    m2b_colores = ['lawngreen', 'blueviolet']
    m2b_estilos = ['-', ':']
    Subredes['M2b Markers'] = {'variables': m2b_vars, 'labels': m2b_labels,
                                 'colores': m2b_colores, 'estilos': m2b_estilos}

    # 7. MACRÓFAGOS M2c
    m2c_vars = ['STAT3', 'IL10OUT']
    m2c_labels = ['STAT3', 'IL-10']
    m2c_colores = ['lawngreen', 'blueviolet']
    m2c_estilos = ['-', ':']
    Subredes['M2c Markers'] = {'variables': m2c_vars, 'labels': m2c_labels,
                                 'colores': m2c_colores, 'estilos': m2c_estilos}

    # 8. CREB Y GSK3B
    creb_vars = ['CREB1', 'GSK3B', 'AKT1', 'AKT3']
    creb_labels = ['CREB', 'GSK3B', 'AKT1', 'AKT3']
    creb_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    creb_estilos = ['-', ':', '--', '-.']
    Subredes['Crebs cycle and Gsk3b'] = {'variables': creb_vars, 'labels': creb_labels,
                             'colores': creb_colores, 'estilos': creb_estilos}

    # 9. Secreted cytokines
    Sc_vars = ['IL6OUT', 'IL12OUT', 'IL18OUT', 'IL33OUT', 'IL10OUT']
    Sc_labels = ['IL6OUT', 'IL12OUT', 'IL18OUT', 'IL33OUT', 'IL10OUT']
    Sc_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red', 'grey']
    Sc_estilos = ['-', ':', '--', '-.','-']
    Subredes['Secreted Macrophage Cytokines'] = {'variables': Sc_vars, 'labels': Sc_labels,
                              'colores': Sc_colores, 'estilos': Sc_estilos}

    # 10. FENOTIPOS DE CÉLULAS T
    t_fenotipos_vars = ['TBET', 'GATA3', 'RORGT', 'FOXP3', 'BCL6']
    t_fenotipos_labels = ['Th1', 'Th2', 'Th17', 'Treg', 'Tfh']
    t_fenotipos_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red', 'grey']
    t_fenotipos_estilos = ['', '', '', '', '']
    Subredes['Phenotypes'] = {'variables': t_fenotipos_vars, 'labels': t_fenotipos_labels,
                              'colores': t_fenotipos_colores, 'estilos': t_fenotipos_estilos}

    # 10. PRESENTACIÓN DE ANTÍGENO
    antigeno_vars = ['TCR', 'CD28', 'CTLA4', 'NDRG1']
    antigeno_labels = ['TCR', 'CD28', 'CTLA4', 'NDRG1']
    antigeno_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    antigeno_estilos = ['-', ':', '--', '-.']
    Subredes['Antigen presentation'] = {'variables': antigeno_vars, 'labels': antigeno_labels,
                                        'colores': antigeno_colores, 'estilos': antigeno_estilos}

    # 11. METABOLISMO DE CÉLULAS T
    t_metabolismo_vars = ['Glycolysis', 'OXPHOS', 'AMPATPratio', 'AMPK']
    t_metabolismo_labels = ['Glycolysis', 'OXPHOS', 'AMP/ATP ratio', 'AMPK']
    t_metabolismo_colores = ['lawngreen', 'blueviolet', 'tab:cyan', 'tab:red']
    t_metabolismo_estilos = ['-', ':', '--', '-.']
    Subredes['T cells metabolism'] = {'variables': t_metabolismo_vars, 'labels': t_metabolismo_labels,
                                'colores': t_metabolismo_colores, 'estilos': t_metabolismo_estilos}

    # 12. FACTORES DE TRANSCRIPCIÓN DE CÉLULAS T
    t_transcripcion_vars = ['AP1', 'NFAT', 'NFKB']
    t_transcripcion_labels = ['AP1', 'NFAT', 'NFKB']
    t_transcripcion_colores = ['lawngreen', 'blueviolet', 'tab:cyan']
    t_transcripcion_estilos = ['-', ':', '--']
    Subredes['T cells FT'] = {'variables': t_transcripcion_vars, 'labels': t_transcripcion_labels,
                                  'colores': t_transcripcion_colores, 'estilos': t_transcripcion_estilos}

    # 13. MARCADORES DE ACTIVACIÓN
    activacion_vars = ['IL2G', 'MTORC1', 'MTORC2']
    activacion_labels = ['IL2G', 'MTORC1', 'MTORC2']
    activacion_colores = ['lawngreen', 'blueviolet', 'tab:cyan']
    activacion_estilos = ['-', ':', '--']
    Subredes['Activation markers'] = {'variables': activacion_vars, 'labels': activacion_labels,
                                        'colores': activacion_colores, 'estilos': activacion_estilos}

    # 14. CÉLULAS Th1
    th1_vars = ['TBET', 'IFNG']
    th1_labels = ['TBET', 'IFNG']
    th1_colores = ['lawngreen', 'blueviolet']
    th1_estilos = ['-', ':']
    Subredes['Th1'] = {'variables': th1_vars, 'labels': th1_labels,
                      'colores': th1_colores, 'estilos': th1_estilos}

    # 15. CÉLULAS Th2
    th2_vars = ['GATA3', 'IL4']
    th2_labels = ['GATA3', 'IL4']
    th2_colores = ['lawngreen', 'blueviolet']
    th2_estilos = ['-', ':']
    Subredes['Th2'] = {'variables': th2_vars, 'labels': th2_labels,
                      'colores': th2_colores, 'estilos': th2_estilos}

    # 16. CÉLULAS Th17
    th17_vars = ['IL17', 'IL21', 'RORGT']
    th17_labels = ['IL17', 'II21', 'RORGT']
    th17_colores = ['lawngreen', 'blueviolet', 'tab:cyan']
    th17_estilos = ['-', ':', '--']
    Subredes['Th17'] = {'variables': th17_vars, 'labels': th17_labels,
                       'colores': th17_colores, 'estilos': th17_estilos}

    # 17. CÉLULAS Treg
    treg_vars = ['FOXP3', 'TGFB', 'CTLA4DIM']
    treg_labels = ['FOXP3', 'TGFB', 'CTLA4DIM']
    treg_colores = ['lawngreen', 'blueviolet', 'tab:red']
    treg_estilos = ['-', ':', '--']
    Subredes['Treg'] = {'variables': treg_vars, 'labels': treg_labels,
                       'colores': treg_colores, 'estilos': treg_estilos}

    # 18. CÉLULAS Tfh
    tfh_vars = ['BCL6', 'IL21', 'CD40L', 'IL9']
    tfh_labels = ['BCL6', 'IL21', 'CD40L', 'IL9']
    tfh_colores = ['lawngreen','tab:red' , 'tab:cyan', 'blueviolet']
    tfh_estilos = ['-', ':', '--', '-.']
    Subredes['Tfh'] = {'variables': tfh_vars, 'labels': tfh_labels,
                      'colores': tfh_colores, 'estilos': tfh_estilos}

    network = NetworkInterface(Subredes)
