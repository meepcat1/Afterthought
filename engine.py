import helpful_functions as hf
import numpy as np
import matplotlib.pyplot as plt
from standard_atmosphere import atmosphere


class Engine:
    def __init__(self, oxidizer, fuel, F_t, opt_alt=0.0, CR=None):

        self.ceaObj = hf.CEA_create(oxidizer=oxidizer, fuel=fuel, CR=CR)
        self.F_t = F_t
        self.P_a, _, _ = atmosphere(opt_alt)
        self.P_a = self.P_a / 1000.0
        self.kPa_psi = 6.89476
        self.lbf_N = 4.44822
        

    #evaluate Cstar at a pressure (kPa) and MR, in m/s
    def eval_cstar(self, P_c, MR):
        return self.ceaObj.get_Cstar(P_c, MR)
        
    def thrust_coeff(self, P_c, MR, eps=40.0, frozenAtThroat=0):
        _, val, _ = self.ceaObj.getFrozen_PambCf(self.P_a, P_c, MR, eps, frozenAtThroat)
        return val

    def chamber_vals(self, P_c, MR, L_star, L_c, cont_angle, cstar_correction=1.0, c_F_correction=1.0, eps='select', frozen=0, frozenAtThroat=0, eps_correction=1.0):
        if eps == 'select':
            eps = self.ceaObj.get_eps_at_PcOvPe(P_c, MR, P_c/(eps_correction*self.P_a), frozen, frozenAtThroat)            

        self.cstar = cstar_correction * self.ceaObj.get_Cstar(P_c, MR)
        self.c_F = c_F_correction * self.thrust_coeff(P_c, MR, eps)
        print(f'Thrust: {self.F_t} lbf')
        print(f'Coeffecient of Thrust: {self.c_F}')
        print(f'Characteristic Velocity: {self.cstar} m/s')
        self.Isp = self.c_F * self.cstar / 9.8065
        self.m_dot = self.F_t * self.lbf_N / (self.Isp * 9.8065)
        print(f'ISP: {self.Isp} s')
        print(f'Mass Flow: {self.m_dot} kg/s')
        self.A_t = (self.cstar * self.m_dot) / (P_c * 1000)
        print(f'Throat Area: {self.A_t} m^2')
        print(f'Expansion Ratio: {eps}')
        
        self.f_dot = self.m_dot/(1+MR)
        self.o_dot = self.m_dot - self.f_dot
        print(f'Fuel Mass Flow: {self.f_dot} kg/s')
        print(f'Oxygen Mass Flow: {self.o_dot} kg/s')
        self.L_star = L_star
        self.flame_temp = self.ceaObj.get_Tcomb(P_c, MR)
        V_c = self.L_star * self.A_t
        print(f'Throat Diameter: {39.3701*2*np.sqrt(self.A_t/np.pi)} in')
        print(f'Exit Diameter: {39.3701*2*np.sqrt((self.A_t*eps)/np.pi)} in')
        print(f'Chamber Length: {L_c} m')
        Contraction_angle = cont_angle

        #from huzel huang pg 74, with chamber length determined from figure 4-10. 
        contraction_ratio, _ = (hf.vol_solver(self.A_t, L_c, Contraction_angle, V_c))
        print(f'Contraction Ratio: {contraction_ratio}')
        print(f'Chamber Diameter: {39.3701*2*np.sqrt((self.A_t*contraction_ratio)/np.pi)} in')



    def chamber_temp_plot(self, mr_range, P_c, eps=40, frozen=0, frozenAtThroat=0):
        mrs = np.linspace(*mr_range, num=1000)
        temps = np.zeros((3,len(mrs)))     
        labels = ['Chamber', 'Throat', 'Exit']
        for j,mr in enumerate(mrs):
            temps[:,j] = self.ceaObj.get_Temperatures(Pc=P_c, MR=mr, eps=eps, frozen=frozen, frozenAtThroat=frozenAtThroat)
        for i in range(3):
            plt.plot(mrs, temps[i,:], label = f'Location = {labels[i]}')
        
        plt.legend(loc='best')
        plt.grid(True)
        plt.title( self.ceaObj.desc)
        plt.xlabel('Mixture Ratio')
        plt.ylabel('Temperature [K]')
        plt.show()
        

    def plot_ISP_Cstar(self, P_c, MR_range, cstar_correction=1.0, c_F_correction=1.0, eps=40.0, frozen=0, frozenAtThroat=0):
        mrs = np.linspace(*MR_range, num=1000)
        vals = np.zeros((2,len(mrs)))

        for j,mr in enumerate(mrs):
            vals[0,j] = cstar_correction * self.ceaObj.get_Cstar(P_c, mr)
            vals[1,j] = c_F_correction * self.thrust_coeff(P_c, mr, eps) * cstar_correction * self.ceaObj.get_Cstar(P_c, mr) / 9.8065
        
        fig, ax1 = plt.subplots()
        ax1.plot(mrs, vals[1,:], 'b-')
        ax1.set_xlabel('Mixture Ratio')
        ax1.set_ylabel('Specific Impulse [s]', color='b')
        ax1.tick_params(axis='y', labelcolor='b')


        ax2 = ax1.twinx()
        ax2.plot(mrs, vals[0,:], 'r--')
        ax2.set_ylabel('Cstar [m/s]', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        

        plt.grid(True)
        plt.title( self.ceaObj.desc)
        plt.show()



    def plot_MR_ISP(self, mr_range, PCs, eps=40.0, frozen=0, frozenAtThroat=0):
        mrs = np.linspace(*mr_range, num = 1000)
        ispArr = np.zeros((len(PCs),len(mrs)))
        for i,pc in enumerate(PCs):
            for j,mr in enumerate(mrs):
                ispArr[i,j]=self.ceaObj.get_Isp(Pc=pc, MR=mr, eps=eps, frozen=frozen, frozenAtThroat=frozenAtThroat)
            plt.plot(mrs, ispArr[i,:], label = 'PC=%g kPa'%pc)

        plt.legend(loc='best')
        plt.grid(True)
        plt.title( self.ceaObj.desc)
        plt.xlabel('Mixture Ratio')
        plt.ylabel('Isp [s]')
        plt.show()
    
    def plot_MR_Cstar(self, mr_range, PCs):
        mrs = np.linspace(*mr_range, num = 1000)
        ispArr = np.zeros((len(PCs),len(mrs)))
        for i,pc in enumerate(PCs):
            for j,mr in enumerate(mrs):
                ispArr[i,j]=self.ceaObj.get_Cstar(Pc=pc, MR=mr)
            plt.plot(mrs, ispArr[i,:], label = 'PC=%g kPa'%pc)

        plt.legend(loc='best')
        plt.grid(True)
        plt.title( self.ceaObj.desc)
        plt.xlabel('Mixture Ratio')
        plt.ylabel('Cstar [m/s]')
        plt.show()

    def plot_PC_ISP(self, pc_range, MRs, eps=40.0, frozen=0, frozenAtThroat=0):
        pcs = np.linspace(*pc_range, num = 1000)
        ispArr = np.zeros((len(MRs),len(pcs)))
        for i,mr in enumerate(MRs):
            for j,pc in enumerate(pcs):
                ispArr[i,j]=self.ceaObj.get_Isp(Pc=pc, MR=mr, eps=eps, frozen=frozen, frozenAtThroat=frozenAtThroat)
            plt.plot(pcs, ispArr[i,:], label = 'MR=%g'%mr)

        plt.legend(loc='best')
        plt.grid(True)
        plt.title( self.ceaObj.desc)
        plt.xlabel('P_c [kPa]')
        plt.ylabel('Isp [s]')
        plt.show()


    def plot_PC_Cstar(self, pc_range, MRs):
        pcs = np.linspace(*pc_range, num = 1000)
        ispArr = np.zeros((len(MRs),len(pcs)))
        for i,mr in enumerate(MRs):
            for j,pc in enumerate(pcs):
                ispArr[i,j]=self.ceaObj.get_Cstar(Pc=pc, MR=mr)
            plt.plot(pcs, ispArr[i,:], label = 'MR=%g'%mr)

        plt.legend(loc='best')
        plt.grid(True)
        plt.title( self.ceaObj.desc)
        plt.xlabel('P_c [kPa]')
        plt.ylabel('Isp [s]')
        plt.show()




