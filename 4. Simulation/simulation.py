import Sofa
import Sofa.Core
import math
import numpy as np
import os

# =============================================================================
# LUNG RESPIRATORY SIMULATION — SOFA FRAMEWORK
# Master's Thesis Implementation
#
# PIPELINE OVERVIEW:
#   1. Load patient-specific tetrahedral meshes (right & left lung) from .msh files
#   2. Compute rest volume and geometric properties at initialisation
#   3. Drive breathing motion via Displacement Boundary Conditions (DBC):
#        - Superior-Inferior (SI) : diaphragm descent, 10–25 mm (clamped)
#        - Ventral               : anterior chest wall expansion, ~5 mm
#        - Lateral               : outward rib-cage expansion, ~2 mm
#   4. Animate using a cosine-shaped breathing waveform (1.3 s insp / 2.6 s exp)
#   5. Sample volumetric data per cycle and report clinical metrics:
#        - FRC, end-inspiratory volume, tidal volume, VT/TLC, diaphragm displacement
#   6. Coordinate both lungs via LungCoordinator and log combined results to file
#

# Method   : Displacement Boundary Condition (DBC)
# Target   : VT [400-500]ml |  Diaphragm displacement 10–25 mm
# =============================================================================

_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_results_0100.txt')
_log = open(_log_path, 'w', buffering=1)

def log(msg=''):
    _log.write(str(msg) + '\n')
    _log.flush()

def read_tets_from_msh(filepath):
    """Parse tetrahedral element connectivity from a Gmsh .msh file (element type 4)."""
    tets = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        in_elements = False
        skip = 0
        for line in lines:
            line = line.strip()
            if line == '$Elements':
                in_elements = True; skip = 1; continue
            if line == '$EndElements':
                break
            if not in_elements: continue
            if skip: skip -= 1; continue
            parts = line.split()
            if len(parts) < 2: continue
            if int(parts[1]) == 4:
                n_tags     = int(parts[2])
                node_start = 3 + n_tags
                tets.append([int(parts[node_start + i]) - 1 for i in range(4)])
    except Exception as e:
        log(f'  [read_tets] error: {e}')
    return tets

def compute_volume(positions, tets_np):
    """Total mesh volume in mm³ via scalar triple product over all tetrahedra."""
    a = positions[tets_np[:, 0]]; b = positions[tets_np[:, 1]]
    c = positions[tets_np[:, 2]]; d = positions[tets_np[:, 3]]
    return np.abs(np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a))).sum() / 6.0

class LungCoordinator:
    """
    Collects per-cycle metrics from both lungs and prints a combined report
    once both have submitted data for the same cycle number.
    """
    def __init__(self, max_cycles):
        self.max_cycles  = max_cycles
        self._data       = {}   # cycle_num -> {lung_name: dict}

    def report_cycle(self, cycle_num, lung_name, data: dict):
        if cycle_num not in self._data:
            self._data[cycle_num] = {}
        self._data[cycle_num][lung_name] = data

        if len(self._data[cycle_num]) == 2:
            self._print_combined(cycle_num)

    def _print_combined(self, cycle_num):
        sides   = self._data[cycle_num]
        names   = list(sides.keys())

        ve_total          = sum(s['ve']          for s in sides.values())
        vi_total          = sum(s['vi']          for s in sides.values())
        tlc_total         = sum(s['tlc']         for s in sides.values())
        vt_abs_total      = sum(s['vt_absolute'] for s in sides.values())
        strain_total      = 100.0 * (vi_total - ve_total) / ve_total  if ve_total else 0.0
        vt_tlc_total      = 100.0 * (vi_total - ve_total) / tlc_total if tlc_total else 0.0
        disp_avg          = sum(s['max_disp'] for s in sides.values()) / len(sides)

        vol_ok  = '  OK' if 7.0 <= abs(vt_tlc_total) <= 13.0 else '  WARNING: outside clinical 8-12% target (VT/TLC)'
        disp_ok = '  OK' if 5.0 <= disp_avg <= 30.0           else '  WARNING: outside 5-30mm range'

        log()
        log('=' * 60)
        log(f'  COMBINED LUNGS — Cycle {cycle_num}/{self.max_cycles}')
        log('=' * 60)
        for n in names:
            s = sides[n]
            log(f'  [{n}]')
            log(f'    FRC (end-exp. vol.)  : {s["ve"]/1000:.3f} ml')
            log(f'    End-insp. volume     : {s["vi"]/1000:.3f} ml')
            log(f'    Tidal volume (abs.)  : {s["vt_absolute"]:.3f} ml')
            log(f'    Diaphragm disp. (max): {s["max_disp"]:.2f} mm')
        log('  ' + '-' * 56)
        log(f'  TOTALS (both lungs combined)')
        log(f'    End-expiration volume (FRC)      : {ve_total/1000:.3f} ml')
        log(f'    End-inspiration volume           : {vi_total/1000:.3f} ml')
        log(f'    Estimated TLC                    : {tlc_total/1000:.3f} ml')
        log(f'    Tidal volume (absolute)          : {vt_abs_total:.3f} ml')
        log(f'    Strain (VT/FRC)                  : {strain_total:+.2f}%')
        log(f'    True tidal volume (VT/TLC)       : {vt_tlc_total:+.2f}%{vol_ok}')
        log(f'    Avg diaphragm displacement       : {disp_avg:.2f} mm{disp_ok}')
        log('=' * 60)
        log()

        if cycle_num == self.max_cycles:
            log('  VALIDATION COMPLETE — Both Lungs Combined')
            log(f'  {self.max_cycles} cycles measured')
            log(f'  Method   : Displacement Boundary Condition (DBC)')
            log('=' * 60)
            log()


def createScene(rootNode):

    rootNode.gravity = [0, 0, 0]
    rootNode.dt      = 0.02

    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('VisualStyle',
        displayFlags='showVisualModels hideWireframe hideBehaviorModels hideCollisionModels')

    for plugin in [
        'Sofa.Component.IO.Mesh',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
    ]:
        rootNode.addObject('RequiredPlugin', name=plugin)

    rootNode.addObject('LightManager')
    rootNode.addObject('DirectionalLight',
        name='light', direction=[0, -1, -0.5], color=[1, 1, 1])

    coordinator = LungCoordinator(max_cycles=7)

    def addLung(name, filename, color):

        scene_dir = os.path.dirname(os.path.abspath(__file__))
        mesh_path = os.path.join(scene_dir, filename)
        tets      = np.array(read_tets_from_msh(mesh_path), dtype=np.int32)
        log(f'[{name}] Read {len(tets)} tetrahedra from {filename}')

        node = rootNode.addChild(name)
        node.addObject('MeshGmshLoader',   name='loader', filename=filename, createSubelements=False)
        node.addObject('TetrahedronSetTopologyContainer',  name='topo', src='@loader')
        node.addObject('TetrahedronSetTopologyModifier')
        node.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')
        node.addObject('MechanicalObject', name='dofs', template='Vec3d', src='@loader')

        visual = node.addChild('Visual')
        visual.addObject('OglModel',           name='visualModel', src='@../loader', color=color)
        visual.addObject('BarycentricMapping', input='@../dofs',   output='@visualModel')

        class BreathingController(Sofa.Core.Controller):

            def __init__(self, lung_node, tets_data, coordinator, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lung_node   = lung_node
                self.tets        = tets_data
                self.coordinator = coordinator
                self.dofs      = None
                self.rest_pos  = None
                self.centroid  = None
                self.w         = None

                # 1:2 inspiration-to-expiration ratio
                self.t        = 0.0
                self.t_insp   = 1.3
                self.t_exp    = 2.6
                self.t_cycle  = 3.9

                self.d_ventral = 5.0      # anterior expansion (mm)
                self.d_lateral = 2.0      # lateral expansion (mm)

                self.vol_ref        = None
                self.cycle_count    = 0
                self.max_cycles     = 7
                self.max_disp_cycle = 0.0
                self.cycle_vol_insp = None
                self.base_z_ref     = None
                self.next_cycle_t   = self.t_cycle

            def onSimulationInitDoneEvent(self, event):
                self.dofs = self.lung_node.getObject('dofs')
                if self.dofs is None:
                    log(f'[{name}] ERROR: dofs not found'); return

                pos           = np.array([list(p) for p in self.dofs.position.value])
                self.rest_pos = pos.copy()
                self.centroid = pos.mean(axis=0)

                zmin, zmax  = pos[:, 2].min(), pos[:, 2].max()
                lung_height = zmax - zmin

                # SI amplitude scaled from lung height, clamped to 10–25 mm
                d_SI_raw  = lung_height * 0.13
                self.d_SI = min(25.0, max(10.0, d_SI_raw))

                # Linear weight: 1 at base (diaphragm), 0 at apex
                norm_z = (pos[:, 2] - zmin) / (zmax - zmin)
                self.w = 1.0 - norm_z

                # Mean z of the basal 7% of nodes — used as diaphragm reference
                self.base_z_ref = pos[pos[:, 2] <= zmin + lung_height * 0.07, 2].mean()

                if len(self.tets) > 0:
                    self.vol_ref = compute_volume(pos, self.tets)

                log('=' * 60)
                log(f'  LUNG SIMULATION REPORT — {name}')
                log(f'  Patient  : LIDC-IDRI-0164')
                log(f'  Mesh     : {filename}')
                log('=' * 60)
                log(f'  Nodes            : {len(pos)}')
                log(f'  Tetrahedra       : {len(self.tets)}')
                log(f'  Rest volume      : {self.vol_ref/1e3:.3f} ml' if self.vol_ref else '  Rest volume      : N/A')
                log(f'  Breath rate      : {60/self.t_cycle:.1f} breaths/min')
                log(f'  Cycle timing     : {self.t_insp}s insp / {self.t_exp}s exp')
                log(f'  SI amplitude     : {self.d_SI:.1f} mm (clamped to 10-25mm range)')
                log(f'  Ventral amplitude: {self.d_ventral} mm')
                log(f'  Lateral amplitude: {self.d_lateral} mm (outward)')
                log('=' * 60)
                log()

            def _phi(self, t):
                """Cosine waveform returning 0 at end-expiration and 1 at peak inspiration."""
                tc = t % self.t_cycle
                if tc < self.t_insp:
                    return 0.5 * (1.0 - math.cos(math.pi * tc / self.t_insp))
                else:
                    te = tc - self.t_insp
                    return 0.5 * (1.0 + math.cos(math.pi * te / self.t_exp))

            def onAnimateBeginEvent(self, event):
                if self.rest_pos is None:
                    return

                prev_t   = self.t
                self.t  += rootNode.dt.value
                phi      = self._phi(self.t)
                prev_phi = self._phi(prev_t)

                new_pos = self.rest_pos.copy()

                # Superior-Inferior: displace nodes downward along z, weighted by w
                new_pos[:, 2] -= phi * self.d_SI * self.w

                # Anisotropic radial expansion: ventral (y) > lateral (x)
                xy   = self.rest_pos[:, :2] - self.centroid[:2]
                norm = np.linalg.norm(xy, axis=1, keepdims=True)
                norm = np.where(norm < 1e-6, 1.0, norm)
                radial_unit = xy / norm

                aniso_disp       = np.empty_like(radial_unit)
                aniso_disp[:, 0] = self.d_lateral * radial_unit[:, 0]
                aniso_disp[:, 1] = self.d_ventral  * radial_unit[:, 1]

                new_pos[:, :2] += phi * self.w[:, None] * aniso_disp

                with self.dofs.position.writeable() as p:
                    p[:] = new_pos

                if self.cycle_count >= self.max_cycles or self.vol_ref is None:
                    return

                # Track peak diaphragm displacement across the cycle
                zmin0, zmax0 = self.rest_pos[:, 2].min(), self.rest_pos[:, 2].max()
                base_mask    = self.rest_pos[:, 2] <= zmin0 + (zmax0 - zmin0) * 0.07
                disp         = abs(new_pos[base_mask, 2].mean() - self.base_z_ref)
                self.max_disp_cycle = max(self.max_disp_cycle, disp)

                # Sample volume once at peak inspiration (phi crosses 0.98)
                if prev_phi < phi and phi > 0.98 and self.cycle_vol_insp is None:
                    self.cycle_vol_insp = compute_volume(new_pos, self.tets)

                if self.t >= self.next_cycle_t:
                    self.next_cycle_t += self.t_cycle
                    self.cycle_count  += 1

                    vi = self.cycle_vol_insp if self.cycle_vol_insp else self.vol_ref
                    ve = self.vol_ref

                    # Tidal volume in ml (mm³ → ml)
                    vt_absolute = (vi - ve) / 1000.0

                    # VT as a fraction of FRC
                    strain_percent = 100.0 * (vi - ve) / ve

                    # VT as a fraction of TLC; TLC estimated as FRC / 0.5
                    tlc_estimated  = ve / 0.5
                    vt_tlc_percent = 100.0 * (vi - ve) / tlc_estimated

                    self.coordinator.report_cycle(self.cycle_count, name, {
                        've':          ve,
                        'vi':          vi,
                        'tlc':         tlc_estimated,
                        'vt_absolute': vt_absolute,
                        'strain':      strain_percent,
                        'vt_tlc':      vt_tlc_percent,
                        'max_disp':    self.max_disp_cycle,
                    })

                    self.max_disp_cycle = 0.0
                    self.cycle_vol_insp = None

        node.addObject(BreathingController(
            lung_node=node, tets_data=tets, coordinator=coordinator, name='breathCtrl'))
        return node

    addLung('RightLung', 'LIDC-IDRI-0100_right_lung.msh', '0.85 0.2 0.2 0.85')
    addLung('LeftLung',  'LIDC-IDRI-0100_left_lung.msh',  '0.7 0.15 0.15 0.85')
