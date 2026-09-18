import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from LUP_solver import solve_lup

c = 299792458
G = 6.67430e-11
Y = 31556925.19
T0 = 2002.32 * Y
R_BH = 27e3 * Y * c
Rad_to_arcsec = 206264.8
delta_BH_correction = np.cos(-29.0078 / 180 * np.pi)

def F(params):
    '''
    params = [x, y, z, vx, vy, vz, mu]
    '''
    r = params[:3]
    R = np.linalg.norm(r)
    v = params[3:6]
    mu = params[6]
    a = - mu / (c ** 2 * R ** 3) * ( r * (c ** 2 - 4 * mu / R + np.dot(v, v)) - 4 * v * np.dot(v, r) )
    return np.hstack((v, a, 0))

def dFdx(params):
    """
    Вычисляет dF/dx для 3D системы
    F = [vx, vy, vz, ax, ay, az]
    x = [x, y, z, vx, vy, vz]
    """
    r = params[:3]
    R = np.linalg.norm(r)
    mu = params[6]
    
    J = np.zeros((6, 6))
    
    # Производные скоростей
    J[0:3, 3:6] = np.eye(3)
    
    # Производные ускорений 
    I = np.eye(3)
    r_outer = np.outer(r, r)
    J[3:6, 0:3] = - (mu / R ** 5) * (I * R**2 - 3 * r_outer)
    
    return J

def dFdP(params):
    """
    Вычисляет dF/dP для 3D системы
    F = [vx, vy, vz, ax, ay, az]
    P = [x0, y0, z0, vx0, vy0, vz0, mu]
    """
    r = params[:3]
    r_norm = np.linalg.norm(r)
    
    # dFdP имеет размерность 6x7 (6 состояний, 7 параметров: [x, y, z, vx, vy, vz, p])
    dFdP = np.zeros((6, 7))
    
    a_deriv = -r / (r_norm ** 3)
    dFdP[3, 6] = a_deriv[0]  # dax/dp
    dFdP[4, 6] = a_deriv[1]  # day/dp
    dFdP[5, 6] = a_deriv[2]  # daz/dp
    
    return dFdP

def dynamic_law(params_dxdP):
    params = params_dxdP[0]
    dxdP = params_dxdP[1:, :]
    return np.vstack((F(params), dFdP(params) + dFdx(params) @ dxdP))

def g(pos):
    return np.array([-1, 1]) * pos / R_BH * Rad_to_arcsec

def dgdx(pos):
    res = np.zeros((2, 6))
    res[0, 0] = - 1 / R_BH * Rad_to_arcsec
    res[1, 1] = 1 / R_BH * Rad_to_arcsec
    return res

def rk_step(x, h, f):
    k1 = f(x)
    k2 = f(x + h / 2 * k1)
    k3 = f(x + h / 2 * k2)
    k4 = f(x + h * k3)
    x_new = x + h * (1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4)
    return x_new


def display_3D(coords_list):
    """
    Отображает траекторию движения в 3D пространстве
    coords_list - массив формы (m, n, 3) с координатами [x, y, z]
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0], [0], [0], c='red')

    for coords in coords_list:
        # Извлекаем координаты
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        z_coords = coords[:, 2]
    
        # Создаем 3D график
        ax.scatter(x_coords, y_coords, z_coords, s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Траектория движения в 3D')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def display_2D(obs):
    plt.scatter(obs[:, 0], obs[:, 1], 1)
    plt.scatter(0, 0, color='r')
    plt.axis('equal')
    plt.show()

def gen_obs(params, t_obs):
    h = 0.001 * Y
    obs = np.zeros((len(t_obs), 2))
    current_time = T0
    current_state = params.copy()
    
    for i in range(len(t_obs)):
        while current_time < t_obs[i]:
            dt = min(h, t_obs[i] - current_time)
            current_state = rk_step(current_state, dt, F)
            current_time += dt

        position = current_state[:2] # Координаты x y звезды
        obs[i] = g(position)
        
    return obs

def error(params, obs, t_obs):
    true_obs = gen_obs(params, t_obs)
    return np.linalg.norm((true_obs - obs).flatten())

def gen_trace(params, t):
    h = 0.001 * Y
    trace = np.zeros((len(t), 3))
    current_time = T0
    current_state = params.copy()
    
    for i in range(len(t)):
        while current_time < t[i]:
            dt = min(h, t[i] - current_time)
            current_state = rk_step(current_state, dt, F)
            current_time += dt

        trace[i] = current_state[:3] # Координаты x y z звезды
        
    return trace

class gauss_newton:    
    def __init__(self, dynamic_law, params_dim, state_dim, obs_dim, h, accuracy=1e-4):
        self.dynamic_law = dynamic_law
        self.params_dim = params_dim    # размерность вектора параметров (state, mu)
        self.state_dim = state_dim      # размерность вектора состояния (x, y, z, vx, vy, vz)
        self.obs_dim = obs_dim          # размерность наблюдаемых данных (alhpa, delta)

        self.h = h
        self.accuracy = accuracy

    def calc_r_A(self, params, obs, t_obs):

        r = np.zeros(self.obs_dim * len(t_obs))
        A = np.zeros((len(t_obs) * self.obs_dim, self.params_dim))
        
        dxdP = np.eye(self.state_dim, self.params_dim)
        current_state = np.vstack((params.copy(), dxdP))
        current_time = T0
        
        for i in range(len(t_obs)):
            #print(f"Наблюдение {i + 1} из {len(t_obs)}")
            while current_time < t_obs[i]:
                dt = min(self.h, t_obs[i] - current_time)
                current_state = rk_step(current_state, dt, self.dynamic_law)
                current_time += dt

            position = current_state[0, :2] # Координаты x y звезды
            dxdP = current_state[1:, :]
            r[[2*i, 2*i + 1]] = obs[i] - g(position)
            A[2*i, :] = - dgdx(position)[0] @ dxdP
            A[2*i + 1, :] = - dgdx(position)[1] @ dxdP

        return r, A
    
    def params_next(self, params, obs, t_obs):
        """Один шаг метода Гаусса-Ньютона"""
        r, A = self.calc_r_A(params, obs, t_obs)
        A_T = A.transpose()
        
        G = A_T @ A
        Ar = A_T @ r

        selected = np.array([0, 1, 6])  # Индексы по которым уточняется система
        params_step_selected = solve_lup(G[np.ix_(selected, selected)], Ar[selected])
        params_step = np.zeros(self.params_dim)
        params_step[selected] = params_step_selected
        return params - params_step
    
    def diff_params(self, params_prev, params_cur):
        return np.linalg.norm(params_prev - params_cur) / np.linalg.norm(params_prev)
    
    def fit(self, t_obs, obs, init_params, max_iterations=1000):
        """
        t_obs: массив времен наблюдений
        obs: наблюдаемые траектории (obs_num, obs_dim) 
        init_params: начальное приближение [x0, y0, z0, vx0, vy0, vz0, mu0]
        max_iterations: максимальное количество итераций
        """
        
        r = error(init_params, obs, t_obs)
        print(f"Ошибка: {r:.6f}")
        
        params = init_params.copy()
        params_prev = params.copy()
        params = self.params_next(params, obs, t_obs)
        self.iteration = 1
        r = error(params, obs, t_obs)
        print(f"Итерация {self.iteration}, ошибка: {r:.6f}")
        
        while (self.diff_params(params_prev, params) > self.accuracy and 
               self.iteration < max_iterations):
            params_prev = params.copy()
            params = self.params_next(params, obs, t_obs)
            self.iteration += 1
            
            # Вывод прогресса
            if self.iteration % 1 == 0:
                r = error(params, obs, t_obs)
                print(f"Итерация {self.iteration}, ошибка: {r:.6f}")
        
        return params
    

if __name__ == "__main__":

    init_data = np.genfromtxt("S2 initial parameters.txt")  # [x0, y0, z0, vx0, vy0, vz0, mu, Tp]
    params = init_data[:7]
    T0 = init_data[7] * Y
    
    obs_data = np.genfromtxt("S2 observations.txt")
    t_obs = obs_data[:, 0] * Y
    obs = obs_data[:, [2, 1]]

    y = gen_obs(params, t_obs)
    plt.scatter(obs[:, 0], obs[:, 1], 2, color='r')
    plt.scatter(y[:, 0], y[:, 1], 2, color='b')
    plt.show()

    gn = gauss_newton(dynamic_law, params_dim=7, state_dim=6, obs_dim=2, h=0.001 * Y)
  
    # Начальное приближение 
    init_params = params.copy()
    print("Начальное приближение: ", init_params)
    print("Отклонение начального приближения: ", (init_params - params) / params)
    
    estimated_params = gn.fit(t_obs, obs, init_params, max_iterations=100)
    
    print("Истинные параметры: \n", params)
    print("Оцененные параметры: \n", estimated_params)
    print("Количество итераций: ", gn.iteration)
    
    accuracy = np.linalg.norm(estimated_params - params) / np.linalg.norm(params)
    print("Относительная ошибка оценки параметров: ", accuracy)
    

