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
    A class that sets all important parameters of a simple compressible convection problem.
    """

    def __init__(self, de_domain, atmosphere, Ra, Pr, **kwargs):
        """ Initializes the convective experiment.

        Parameters
        ----------
        de_domain       : A DedalusDomain object
            Contains info about the dedalus domain
        kwargs          : Dictionary
            Additional keyword arguments for the BoussinesqConvection._set_parameters function
        """
        self.de_domain  = de_domain
        self.atmosphere = atmosphere
        self._set_diffusivities(Ra, Pr, **kwargs)
        return

    def _set_diffusivities(self, Ra, Pr, const_diffs=False):
        """
        Set up important parameters of the problem for boussinesq convection. 

        Parameters
        ----------
        Rayleigh        : Float 
            The Rayleigh number, as defined in Anders, Brown, Oishi 2018 and elsewhere
        Prandtl         : Float
            The Prandtl number, viscous / thermal diffusivity
        IH              : bool
            If True, internally heated convection. If false, boundary-driven convection.

        """
        Lz, delta_s, Cp, g = [self.atmosphere.atmo_params[k] for k in ('Lz', 'delta_s', 'Cp', 'g')]
        nu_top  = np.sqrt(Pr * g * Lz**3 * np.abs(delta_s/Cp) / Ra)
        chi_top = nu_top/Pr

        if const_diffs:
            self.atmosphere.atmo_params['chi0'] = chi_top
            self.atmosphere.atmo_params['nu0']  = nu_top
            t_therm = Lz**2/chi_top
        else:
            self.atmosphere.atmo_fields['chi0']['g'] = chi_top/self.atmosphere.atmo_fields['rho0']['g']
            self.atmosphere.atmo_fields['nu0']['g']  =  nu_top/self.atmosphere.atmo_fields['rho0']['g']
            t_therm = Lz**2/np.mean(self.atmosphere.atmo_fields['chi0'].interpolate(z=Lz/2)['g'])
            [self.atmosphere.atmo_fields[k].set_scales(1, keep_data=True)  for k in ('chi0', 'nu0', 'rho0')]
        self.atmosphere.atmo_params['t_therm'] = t_therm
        logger.info('Experiment set with top of atmosphere chi = {:.2e}, nu = {:.2e}'.format(chi_top, nu_top))
        logger.info('Experimental t_therm/t_buoy = {:.2e}'.format(t_therm/self.atmosphere.atmo_params['t_buoy']))

    def set_IC(self, solver, noise_scale, checkpoint, A0=1e-6, restart=None, checkpoint_dt=1800, overwrite=False, **kwargs):
        """
        Set initial conditions as random noise in the temperature perturbations, tapered to
        zero at the boundaries.  

        Parameters
        ----------
        noise_scale     : NumPy array, size matches local dealiased z-grid.
            The scale of expected thermo pertrubations in the problem (epsilon)
        checkpoint      : A Checkpoint object
            The checkpointing object of the current simulations
        A0              : Float, optional
            The size of the initial perturbation compared to noise_scale. Generally should be very small.
        restart         : String, optional
            If not None, the path to the checkpoint file to restart the simulation from.
        checkpoint_dt   : Int, optional
            The amount of wall time, in seconds, between checkpoints (default 30 min)
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
        checkpoint.set_checkpoint(solver, sim_dt=checkpoint_dt, mode=mode)
        return dt, mode
