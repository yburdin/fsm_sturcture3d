import sys
import numpy as np
import json
from pycufsm.fsm import strip
from pycufsm.preprocess import stress_gen
from typing import Optional


class CrossSectionProperties:
    def __init__(self):
        self.area = None
        self.section_center_x = None
        self.section_center_y = None
        self.inertia_moment_x = None
        self.inertia_moment_y = None
        self.inertia_moment_xy = None
        self.alpha_0 = None
        self.inertia_moment_v = None
        self.inertia_moment_u = None
        self.moment_of_area_x = None
        self.moment_of_area_y = None
        self.shear_center_x = None
        self.shear_center_y = None
        self.shear_center_u = None
        self.shear_center_v = None
        self.torsional_constant = None
        self.warping_constant = None
        self.eccentricity_x = 0
        self.eccentricity_y = 0
        self.section_modulus_u_top = None
        self.section_modulus_u_bot = None
        self.section_modulus_v_top = None
        self.section_modulus_v_bot = None
        self.section_modulus_u_max = None
        self.section_modulus_u_min = None
        self.section_modulus_v_max = None
        self.section_modulus_v_min = None
        self.plastic_section_modulus_u = None
        self.plastic_section_modulus_v = None
        self.gyration_radius_x = None
        self.gyration_radius_y = None
        self.gyration_radius_u = None
        self.gyration_radius_v = None


class Material:
    def __init__(self):
        self.E = None
        self.mu = None
        self.G = None
        self.props = None
        self.yield_strength = None
        self.ultimate_strength = None

        self.yield_average = None


class Analysis:
    def __init__(self):
        self.nodes = None
        self.elements = None

        self.bb_diagonal = None

        self.signature_raw = None
        self.signature = None
        self.curve = None
        self.shapes = None

        self.csp = CrossSectionProperties()
        self.material = Material()

        self.lengths = []
        self.critical_moment = None

        self.sc_derivative = None
        self.sc_derivative_2 = None
        self.index_cr = None

        self.length_cr = None
        self.sigma_cr = None

    def import_data(self, path):
        with open(path, 'r') as f:
            data = json.loads(f.read())

        nodes = data["Nodes"]
        nodes = np.concatenate((nodes, np.ones((len(nodes), 4)), np.zeros((len(nodes), 1))), axis=1)

        dim_x = max(nodes[:, 1]) - min(nodes[:, 1])
        dim_y = max(nodes[:, 2]) - min(nodes[:, 2])
        self.bb_diagonal = np.sqrt(dim_x ** 2 + dim_y ** 2)

        self.material.props = np.array(data["Materials"])
        self.nodes = nodes
        self.elements = np.array(data["Elements"])
        self.lengths = np.array(data["Lengths"])

        (self.csp.section_center_x, self.csp.section_center_y, self.csp.shear_center_x, self.csp.shear_center_y,
         self.csp.alpha_0, self.csp.area, self.csp.inertia_moment_x, self.csp.inertia_moment_xy,
         self.csp.inertia_moment_y, self.csp.inertia_moment_u, self.csp.inertia_moment_v,
         ) = [data["Cross section"][x] for x in ['cx', 'cy', 'x0', 'y0', 'phi', 'A',
                                                 'Ixx', 'Ixy', 'Iyy', 'I11', 'I22']]

    @staticmethod
    def moving_average(data, average_range=7):
        if type(data) is not list:
            data = list(data)

        new_data = [data[0]] * ((average_range - 1) // 2) + data + [data[-1]] * ((average_range - 1) // 2)
        result_data = [sum(new_data[n:n + average_range]) / average_range for n in
                       range(len(new_data) - average_range + 1)]
        return result_data

    def cufsm(self, cufsm_type='signature'):
        # No special springs or constraints
        springs = []
        constraints = []

        # Values here correspond to signature curve basis and orthogonal based upon geometry
        gbt_con = {
            'glob': [0],
            'dist': [0],
            'local': [0],
            'other': [0],
            'o_space': 1,
            'couple': 1,
            'orth': 2,
            'norm': 0,
        }

        # Simply-supported boundary conditions
        b_c = 'S-S'

        # For signature curve analysis, only a single array of ones makes sense here
        m_all = np.ones((len(self.lengths), 1))

        # Solve for 10 eigenvalues
        n_eigs = 1

        # Set the section properties for this simple section
        # Normally, these might be calculated by an external package
        sect_props = {
            'cx': self.csp.section_center_x,
            'cy': self.csp.section_center_y,
            'x0': self.csp.shear_center_x,
            'y0': self.csp.shear_center_y,
            'phi': self.csp.alpha_0,
            'A': self.csp.area,
            'Ixx': self.csp.inertia_moment_x,
            'Ixy': self.csp.inertia_moment_xy,
            'Iyy': self.csp.inertia_moment_y,
            'I11': self.csp.inertia_moment_u,
            'I22': self.csp.inertia_moment_v
        }

        # Generate the stress points assuming pure compression
        if cufsm_type == 'signature':
            force_p = sect_props['A'] * 1
            force_m11 = 0
            force_m22 = 0
        elif cufsm_type == 'critical moment':
            force_p = 0
            if sect_props['I11'] > sect_props['I22']:
                force_m11 = 1e6
                force_m22 = 0
            else:
                force_m11 = 0
                force_m22 = 1e6
        else:
            raise Exception('CUFSM type error')

        nodes_p = stress_gen(
            nodes=self.nodes,
            forces={
                'P': force_p,
                'Mxx': 0,
                'Myy': 0,
                'M11': force_m11,
                'M22': force_m22
            },
            sect_props=sect_props,
        )

        # Perform the Finite Strip Method analysis
        if cufsm_type == 'signature':
            self.signature_raw, self.curve, self.shapes = strip(
                props=self.material.props,
                nodes=nodes_p,
                elements=self.elements,
                lengths=self.lengths,
                springs=springs,
                constraints=constraints,
                gbt_con=gbt_con,
                b_c=b_c,
                m_all=m_all,
                n_eigs=n_eigs,
                sect_props=sect_props
            )
        elif cufsm_type == 'critical moment':
            self.critical_moment, _, _ = strip(
                props=self.material.props,
                nodes=nodes_p,
                elements=self.elements,
                lengths=self.lengths,
                springs=springs,
                constraints=constraints,
                gbt_con=gbt_con,
                b_c=b_c,
                m_all=m_all,
                n_eigs=n_eigs,
                sect_props=sect_props
            )

            self.critical_moment = np.array(self.critical_moment) * 1e6
        else:
            raise Exception('CUFSM type error')

    def critical_stresses(self):
        # Усреднение сигнатурной кривой
        self.signature = self.moving_average(self.signature_raw)

        # Расчет производных
        self.sc_derivative = []
        for i in range(len(self.lengths) - 1):
            self.sc_derivative.append((self.signature[i + 1] - self.signature[i]) /
                                      (self.lengths[i + 1] - self.lengths[i]))

        self.sc_derivative_2 = []
        for i in range(len(self.sc_derivative) - 1):
            self.sc_derivative_2.append((self.sc_derivative[i + 1] - self.sc_derivative[i]) /
                                        (self.lengths[i + 1] - self.lengths[i]))

        # Определение минимумов на сигнатурной кривой
        self.index_cr = []

        # Поиск локальной потери устойчивости
        # ... по первой производной
        local_buckling_index = self.find_local_buckling_index_with_first_derivative(
            max_length=1.5 * self.bb_diagonal)
        if local_buckling_index:
            self.index_cr.append(local_buckling_index)

        # ... по второй производной
        if len(self.index_cr) == 0:
            for i in range(len(self.sc_derivative_2) - 1):
                if (np.sign(self.sc_derivative_2[i]) != np.sign(self.sc_derivative_2[i + 1]) and
                        np.sign(self.sc_derivative_2[i]) == 1 and self.lengths[i] < 1.5 * self.bb_diagonal):
                    self.index_cr.append(i + 1)
                    break

        # Поиск дисторсионной потери устойчивости
        if len(self.index_cr) == 1:
            index_of_max = [i for i in range(len(self.sc_derivative) - 1)
                            if (np.sign(self.sc_derivative[i]) != np.sign(self.sc_derivative[i + 1]) and
                                np.sign(self.sc_derivative[i]) == 1)]

            if len(index_of_max) > 0:
                index_of_max = index_of_max[0]
            else:
                index_of_max = len(self.signature) - 3

            # ... по первой производной
            for i in range(self.index_cr[0], len(self.sc_derivative) - 1):
                if (np.sign(self.sc_derivative[i]) != np.sign(self.sc_derivative[i + 1]) and
                        np.sign(self.sc_derivative[i]) == -1 and self.lengths[i] < 20 * self.bb_diagonal):
                    self.index_cr.append(i + 1)
                    break

            # ... по второй производной
            if len(self.index_cr) == 1:
                for i in range(self.index_cr[0], index_of_max):
                    if (np.sign(self.sc_derivative_2[i]) != np.sign(self.sc_derivative_2[i + 1]) and
                            np.sign(self.sc_derivative_2[i]) == -1 and self.lengths[i] < 20 * self.bb_diagonal):
                        self.index_cr.append(i + 1)
                        break

            if len(self.index_cr) == 1:
                for i in range(index_of_max, len(self.sc_derivative_2) - 1):
                    if (np.sign(self.sc_derivative_2[i]) != np.sign(self.sc_derivative_2[i + 1]) and
                            np.sign(self.sc_derivative_2[i]) == 1 and self.lengths[i] < 20 * self.bb_diagonal):
                        self.index_cr.append(i + 1)
                        break

        # Дополнительная попытка найти критическое напряжение по первой производной с увеличением предельной длины
        if len(self.index_cr) < 1:
            for coefficient in np.linspace(1.6, 19.6, 19):
                local_buckling_index = self.find_local_buckling_index_with_first_derivative(
                    max_length=coefficient * self.bb_diagonal)
                if local_buckling_index:
                    self.index_cr.append(local_buckling_index)
                    break

        self.length_cr = [self.lengths[i] for i in self.index_cr]
        self.sigma_cr = [self.signature[i] for i in self.index_cr]

    def export_results(self, path):
        with open(path, 'w') as f:
            f.write(json.dumps(
                {'Lengths': list(self.lengths),
                 'Signature curve': list(self.signature),
                 'Signature curve unaveraged': list(self.signature_raw),
                 'Critical moment curve': list(self.critical_moment),
                 'Shapes': [shape.reshape(1, -1)[0].tolist() for shape in self.shapes],
                 'Critical stress': list(self.sigma_cr),
                 'Critical length': list(self.length_cr),
                 'Critical indices': list(self.index_cr),
                 },
                indent=4,
            ))

    def find_local_buckling_index_with_first_derivative(
            self,
            max_length: float
    ) -> Optional[int]:
        index_cr = None
        for i in range(len(self.sc_derivative) - 1):
            if (np.sign(self.sc_derivative[i]) != np.sign(self.sc_derivative[i + 1]) and
                    np.sign(self.sc_derivative[i]) == -1 and self.lengths[i] < max_length):
                index_cr = i + 1
                break

        return index_cr


if __name__ == "__main__":
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]

        analysis = Analysis()

        analysis.import_data(input_path)
        print('Import - OK')

        analysis.cufsm()
        analysis.critical_stresses()
        analysis.cufsm(cufsm_type='critical moment')
        print('CUFSM - OK')

        analysis.export_results(output_path)
        print('Export - OK')

    except IndexError:
        print('No path provided')
