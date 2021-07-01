import math

class Pulse:
    def __init__(self):
        self.gamma = 26.75 #constant
        self.type = "ge"
        self.Gtheta = math.pi/2 # 90 degree,
        self.Gphi = 0 # pulse
        self.dt = 0.2 # time step
        self.TE = 30
        self.B0 = 3e4 #field strength

