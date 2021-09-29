

import numpy as np
from sympy import S
from sympy.physics.wigner import wigner_6j,wigner_3j
import sympy as sym

from numpy import e,pi,sqrt,log,log10

h=6.62607015e-34 ## J s
hbar=h/(2*pi)
c=299792458     ## m/s
kB=1.380649e-23 ## J/K
epsilon0=8.8541878128e-12 ## F/m
q=1.602176634e-19 ## electric charge [C]
a0=5.29177210903e-11 ## Bohr radius [m]
uB=9.2740100783e-24 ## bohr magneton J/T

m_cs=2.20694650e-25 ## kg
m_rb=1.44316060e-25 ## kg
class fine_manifold():
    ## Line center
    def __init__(self,I=0,center=0,L=0,S=0,J=0,lifetime=None):
        self.center=center ## Energy, MHz
        self.L=L
        self.S=S
        self.I=I
        self.J=J
        self.lifetime=lifetime ## ns, None means g.s.

        ### Allowable F states:
        ### J+I, J+I-1, ... , |J-I|
        ### To deal with fractions
        self.Fs=[F for F in range(0,int(J+I+1)) if F >= np.abs(J-I) and F<=J+I]
        return

    def __repr__(self):
        string=self.term_symbol(self.L,self.S,self.J)
        string+="\nLine-center={} MHz\n".format(self.center)
        string+="Lifetime={} ns\n".format(self.lifetime)
        for F in self.Fs[::-1]:
            string+=repr(getattr(self,"F"+str(F)))+"\n"
        return string

    def term_symbol(self,L,S,J):
        L_list=["S","P","D","F","G","H"]
        return str(2*S+1)+L_list[L]+"_"+str(J)

    def set_hyperfine(self,energies,moments):
        ### energies: list of energy values (MHz) starting with smallest F value
        #labels=list(map(lambda x: "F"+str(x),self.Fs))
        #E_dict=dict(zip(labels,energies))
        #vars(self).update(E_dict)
        hyperfine_dict={}
        for F,energy,moment in zip(self.Fs,energies,moments):
            hyperfine_dict["F"+str(F)]=hyperfine_manifold(F,energy,moment)
        vars(self).update(hyperfine_dict)
        self.levels=[hyperfine_dict[key] for key in sorted(hyperfine_dict.keys())]
        return

class hyperfine_manifold():
    def __init__(self,F,center,mu):
        ## F: half integer, total angular momentum
        ## mu: linear zeeman shift (MHz/G)
        self.center=center
        self.F=F
        self.B_shift=mu
        return
    def __repr__(self):
        return "F={}  {:.1f} MHz  ({:.2f}  MHz/G)".format(self.F,self.center,self.B_shift)

class transition():

    def __init__(self,init,final,dipole_matrix_element,mass,name):
        ## init, final: fine_manifold
        ## dipole_matrix_element: real, positive, reduced matrix element
        ##  units of [e*a0]
        ##  < J || er || J'>
        ## Specific transitions between hyperfine states computed thru clebsch-gordon coefficients
        ## See: wigner-eckhardt theorem
        self.initial=init
        self.final=final
        self.J_JJ=dipole_matrix_element
        self.mass=mass

        self.compute_F_FF()
        self.compute_mF_mFF()
        self.compute_parameters()
        self.name=name
        return

    def __repr__(self):
        ## generate string for F_FF matrix
        FF_list=[ str(FF)+"\t"  for FF in self.final.Fs]
        header="FF=\t"+"".join(FF_list)+"\n"
        lines=str(self.F_FF_angle).split("\n")
        lines[0]+= "     *"+str(self.J_JJ)
        body=""
        for line,label in zip(lines, ["F="+str(F)+"\t" for F in self.final.Fs]):
            body+= label+line+"\n"
        F_FF_string=header+body

        return F_FF_string

    def compute_F_FF(self):
        ## hyperfine reduced matrix elements, <F||eR||FF> satisfy
        ## <F||eR||FF> = <J||er||JJ>  (-1)^(1+FF+I+J) *sqrt((2F'+1)(2J+1)) * (6J)
        ## first compute <F||er||FF>/<J||er|JJ>, which encodes all the angular dependence of the reduced matrix element
        ## then multiply by <J||er||JJ>
        Fs=self.initial.Fs
        FFs=self.final.Fs
        J=self.initial.J
        JJ=self.final.J
        I=self.initial.I
        F_mat=np.zeros((len(Fs),len(FFs)),type(S(1)))
        F_dict={}
        for F,FF in [(F,FF) for F in Fs for FF in FFs]:
            F_FF_angle=sym.simplify((-1)**(1+FF+I+J) *sym.sqrt((2*FF+1)*(2*J+1)) *wigner_6j(J,JJ,1,FF,F,I))
            F_mat[Fs.index(F),FFs.index(FF)]=F_FF_angle
            label="F"+str(F)+"_FF"+str(FF)
            F_dict[label]=float(F_FF_angle)*self.J_JJ
        ## Keep the angular matrix in symbolic form
        self.F_FF_angle=F_mat
        ## cast as float when scaling by the J_JJ reduced dipole operator
        self.F_FF=F_mat.astype(np.float64)*self.J_JJ
        vars(self).update(F_dict)
        return

    def compute_mF_mFF(self):
        ## dipole matrix elements, <F,mF | eR_q | FF,mFF> satisfy
        ## <F,mF| er |FF,mFF> = <F||er||FF> *(-1)^(FF+mF-1) *sqrt(2F+1) (3j)
        ## These matrix elements depend on the initial and final orientation state: mF, mFF
        ##                                             and the polarization of light, q
        Fs=self.initial.Fs
        FFs=self.final.Fs
        dipole_matrix_dict={}
        human_readable={}
        for F,FF in [(F,FF) for F in Fs for FF in FFs]:
            dipole_matrix_dict[(F,FF)]=np.zeros((len(self.m(F)),len(self.m(FF)),3),dtype=type(S(1)))
            for mF,mFF,q in [(mF,mFF,q) for mF in self.m(F) for mFF in self.m(FF) for q in [0,1,-1]]:
                if wigner_3j(FF,1,F,mFF,q,-mF) != 0:
                    #print("mF={},mFF={},q={}".format(mF,mFF,q))
                    mF_angle=self.F_FF_angle[Fs.index(F),FFs.index(FF)]*(-1)**(FF+mF-1)*sym.sqrt(2*F+1)*wigner_3j(FF,1,F,mFF,q,-mF)
                    human_readable[self.label(F,mF,FF,mFF,q)]=float(mF_angle)*self.J_JJ
                    dipole_matrix_dict[(F,FF)][self.m(F).index(mF),self.m(FF).index(mFF),q]=mF_angle
        self.dipole_matrix_dict=dipole_matrix_dict
        vars(self).update(human_readable)
        return
    def m(self,J):
        """
        return a list of length 2J+1: -J,-J+1,...,0,1,...,J
        """
        return [i-J for i in range(0,2*J+1)]
    def label(self,F,mF,FF,mFF,q):
        def repeat_if_neg(num):
            if num<0:
                return 2*str(np.abs(num))
            else:
                return str(num)

        return "F"+str(F)+"_mF"+repeat_if_neg(mF)+"_FF"+str(FF)+"_mFF"+repeat_if_neg(mFF)+"_q"+repeat_if_neg(q)

    def dipole_matrix(self,F,mF,FF,mFF,q):
        """wrapper for handling index arithmetic"""
        return self.dipole_matrix_dict[(F,FF)][self.m(F).index(mF),self.m(FF).index(mFF),q]

    def compute_parameters(self):
        f=np.abs(self.final.center-self.initial.center)*1e6 # [Hz]
        omega=f/(2*pi) # [rad/s]
        self.wavelength=c/(f) ## wavelength in vacuum [m]
        self.k=2*pi/self.wavelength ## wavenumber in vacuum [cycles/m]
        self.linewidth=1/(self.final.lifetime*1e-9) ## [Hz]

        self.vr=hbar*self.k/self.mass ## recoil velocity [m/s]
        self.Er=hbar**2*self.k**2/(2*self.mass *hbar) ## recoil energy [Hz]
        self.Tr=hbar**2*self.k**2/(self.mass*kB) ## recoil temperature [K]
        self.TD=hbar*self.linewidth/(2*kB) ## Doppler temperature [K]
        self.Isat=hbar*omega**3*self.linewidth/(12*pi*c**2) ## saturation intensity [W/m^2]
        self.scat=h*f*self.linewidth/(2*self.Isat) ## scattering cross-section [m^2]
        self.scat=3*self.wavelength**2/(2*pi) ## scattering cross-section [m^2]


        self.a= h *pi*self.linewidth/(2*pi*self.mass*self.wavelength) ## acceleration scale for doppler cooling/MOT [m/s^2]
        self.doppler = 2*pi / (self.linewidth*self.wavelength) ##doppler shift coefficient  [Hz  / m/s]

        self.MOT_magnetic = (self.final.levels[-1].B_shift-self.initial.levels[-1].B_shift)*1e6 /self.linewidth ## MOT Magnetic force coefficient [linewidth/Gauss]
        return
    def print_parameters(self):
        print("wavelength: {:0.3f} nm".format(self.wavelength*1e9))
        print("wavenumber: {:0.3f} /um".format(self.k*1e-6))
        print("linewidth: 2*pi {:0.3f} MHz".format(self.linewidth*1e-6/(2*pi)))
        print("recoil velocity: {:0.3f} mm/s".format(self.vr*1e3))
        print("recoil energy: 2*pi {:0.3f} kHz".format(self.Er*1e-3/(2*pi)))
        print("recoil temperature: {:0.3f} uK".format(self.Tr*1e6))
        print("Doppler temperature: {:0.3f} uK".format(self.TD*1e6))
        print("Saturation intensity: {:0.3f} mW/cm^2".format(self.Isat*1e3*1e-4))
        print("scattering cross-section: {:0.3f} um^2".format(self.scat*1e12))
        print("MOT acceleration scale:{:0.3f} m/s^2".format(self.a))
        print("Doppler coefficient:{:0.3f} Hz/ m/s".format(self.doppler))
        print("MOT magnetic coefficient:{:0.3f} 1/Gauss".format(self.MOT_magnetic))
        return
class cesium():
    def __init__(self):
        self.name="Cesium-133"
        self.I=S(7)/2
        ## All the states we care about have S=1/2
        self.S=S(1)/2
        self.mass=m_cs
        self.fine_states=[]
        self.S12=fine_manifold(I=self.I,L=0,S=self.S,J=S(1)/2)
        self.S12.set_hyperfine([-5170.855370625,4021.776399375],[-0.35,0.35])
        self.fine_states.append(self.S12)

        self.P32=fine_manifold(I=self.I,center=351725718.50,L=1,S=self.S,J=S(3)/2,lifetime=30.473)
        self.P32.set_hyperfine([-339.7128,-188.4885,-12.79851,263.8906],[-0.93,0,0.37,0.56])
        self.fine_states.append(self.P32)

        self.P12=fine_manifold(I=self.I,center=335116048.807,L=1,S=S(1)/2,J=S(1)/2,lifetime=34.894)
        self.P12.set_hyperfine([-656.820,510.860],[-0.12,0.12])
        self.fine_states.append(self.P12)

        self.D2=transition(self.S12,self.P32,4.4786,self.mass,"D2")
        self.D1=transition(self.S12,self.P12,3.1822,self.mass,"D1")

        self.fine_states.sort(key=lambda x: x.center)
    def __repr__(self):
        string=self.name+"\n"*2
        for state in self.fine_states[::-1]:
            string+= repr(state)+"\n"*5
        return string.rstrip("\n")

class rubidium87():
    def __init__(self):
        self.name="Rubidium-87"
        self.I=S(3)/2
        ## All the states we care about have S=1/2
        self.S=S(1)/2
        self.mass=m_rb
        self.fine_states=[]
        self.S12=fine_manifold(I=self.I,L=0,S=self.S,J=S(1)/2)
        self.S12.set_hyperfine([-4271.67663181519,2563.00597908911],[-0.70,0.70])
        self.fine_states.append(self.S12)

        self.P32=fine_manifold(I=self.I,center=384230484.4685,L=1,S=self.S,J=S(3)/2,lifetime=26.24)
        self.P32.set_hyperfine([-302.0738,-229.8518,-72.9113,193.7408],[0,0.93,0.93,0.93])
        self.fine_states.append(self.P32)

        self.P12=fine_manifold(I=self.I,center=377107463.5,L=1,S=S(1)/2,J=S(1)/2,lifetime=27.70)
        self.P12.set_hyperfine([-510.410,306.246],[-0.23,0.23])
        self.fine_states.append(self.P12)

        self.D2=transition(self.S12,self.P32,4.227,self.mass,"D2")
        self.D1=transition(self.S12,self.P12,2.992,self.mass,"D1")

        self.fine_states.sort(key=lambda x: x.center)
    def __repr__(self):
        string=self.name+"\n"*2
        for state in self.fine_states[::-1]:
            string+= repr(state)+"\n"*5
        return string.rstrip("\n")
