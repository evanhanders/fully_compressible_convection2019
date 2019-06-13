import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    from functions import global_noise
except:
    from sys import path
    path.insert(0, './logic')
    from logic.functions import global_noise

class CompressibleConvection:
    """
    A class that sets up important parameters of a compressible convection problem, and
    which sets initial conditions for that problem.

    Attributes:
    -----------
    de_domain : DedalusDomain
        contains info about the Domain in which the experiment is conducted
    atmosphere : IdealGasAtmosphere or other Atmosphere class
        contains info about the atmosphere in which convection occurs.
    """

    def __init__(self, de_domain, atmosphere, *args, **kwargs):
        """ Initializes the convective experiment.

        Parameters
        ----------
        de_domain, atmosphere -- see class-level docstring
        args, kwargs : list, dictionary
            Additional arguments & keyword arguments for the self._set_diffusivities function
        """
        self.de_domain  = de_domain
        self.atmosphere = atmosphere
        self._set_diffusivities(*args, **kwargs)
        return

    def _set_diffusivities(self, Ra, Pr, **kwargs):
        """
        Sets up domain diffusivities.

        Parameters
        ----------
        Ra  : float
            The Rayleigh number at the top of the domain
        Pr  : float
            The Prandtl number at the top of the domain
        """
        Lz, delta_s, Cp, g = [self.atmosphere.atmo_params[k] for k in ('Lz', 'delta_s', 'Cp', 'g')]
        nu_top  = np.sqrt(Pr * g * Lz**3 * np.abs(delta_s/Cp) / Ra)
        chi_top = nu_top/Pr
        self.atmosphere.set_diffusivites(chi_top, nu_top, **kwargs)

    def set_IC(self, solver, noise_scale, checkpoint, A0=1e-6, restart=None, checkpoint_dt=None, overwrite=False, **kwargs):
        """
        Set initial conditions as entropy perturbations which are random noise and tapered to match boundary conditions.

        Parameters
        ----------
        solver  : Dedalus solver object
            The solver in which the problem is being solved.
        noise_scale     : NumPy array, size matches local dealiased z-grid.
            The scale of expected thermo pertrubations in the problem (epsilon)
        checkpoint      : A Checkpoint object
            The checkpointing object of the current simulations
        A0              : Float, optional
            The size of the initial perturbation compared to noise_scale. Generally should be very small.
        restart         : String, optional
            If not None, the path to the checkpoint file to restart the simulation from.
        checkpoint_dt   : float, optional
            The amount of simulation time between checkpoints
        overwrite       : Bool, optional
            If True, auto-set the file mode to overwrite, even if checkpoint-restarting
        kwargs          : Dict, optional
            Additional keyword arguments for the global_noise() function
        """
        if restart is None:
            # initial conditions
            T1       = solver.state['T1']
            ln_rho1  = solver.state['ln_rho1']
            T1_z     = solver.state['T1_z']
            self.atmosphere.atmo_fields['T0'].set_scales(self.de_domain.dealias, keep_data=True)
            T0_de = self.atmosphere.atmo_fields['T0']['g']
                
            noise = global_noise(self.de_domain, **kwargs)
            noise.set_scales(self.de_domain.dealias, keep_data=True)
            [field.set_scales(self.de_domain.dealias, keep_data=False) for field in (T1, ln_rho1, T1_z)]

            T1['g'] = A0*noise_scale*T0_de*np.sin(np.pi*self.de_domain.z_de/self.atmosphere.atmo_params['Lz'])*noise['g']
            T1.differentiate('z', out=T1_z)
            ln_rho1['g'] = -np.log(T1['g']/T0_de + 1)
            logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0*noise_scale))
            dt = None
            mode = 'overwrite'
            self.atmosphere.atmo_fields['T0'].set_scales(1, keep_data=True)
        else:
            logger.info("restarting from {}".format(restart))
            dt = checkpoint.restart(restart, solver)
            if overwrite:
                mode = 'overwrite'
            else:
                mode = 'append'
        if checkpoint_dt is None: checkpoint_dt = 25*self.atmosphere.atmo_fields['t_buoy']
        checkpoint.set_checkpoint(solver, sim_dt=checkpoint_dt, mode=mode)
        return dt, mode
