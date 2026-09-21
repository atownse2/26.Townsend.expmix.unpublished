import ROOT
import numpy as np
import random
from typing import Any, Optional, Sequence


# Helper functions
def print_par(par: ROOT.RooRealVar):
    """
    Print the parameter name, value, and error.
    """
    print_str = f"{par.GetName()}: {par.getVal()}"
    if hasattr(par, 'isConstant') and not par.isConstant():
        if hasattr(par, 'getErrorHi'):
            print_str += f" + {par.getErrorHi()} {par.getErrorLo()}"
        elif hasattr(par, 'getError'):
            print_str += f" ± {par.getError()}"
        else:
            pass
    
    print(print_str)


# RooFit model wrapper class
class RooFitModel:
    name = "GenericRooFitModel"
    pdf: ROOT.RooAbsPdf

    def __init__(self, *args, **kwargs):
        self.kwargs = dict(kwargs)
        self.prefix = self.kwargs.pop("prefix", "")

        default_specs = getattr(self, "default_par_specs", getattr(self, "par_specs", None))
        assert default_specs is not None, f"Model class {type(self).__name__} must define default_par_specs"
        self.par_specs = dict(default_specs)

        for par_name in list(self.par_specs.keys()):
            if par_name in self.kwargs:
                self.par_specs[par_name] = self.kwargs[par_name]
                self.kwargs.pop(par_name)

        self._parameter_values = {}

        if self.par_specs:
            self.init_params(self.par_specs)

        self.initialize(*args, **self.kwargs)
        if self.kwargs.get("random_initialization", False):
            self.randomize_params(custom_ranges=self.kwargs.get("custom_ranges", {}))

    def initialize(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the initialize method.")

    def _pdf_name(self, default_name: str = "pdf") -> str:
        pdf_name = self.kwargs.get("pdf_name", default_name if default_name is not None else self.name)
        if self.prefix:
            pdf_name = f"{self.prefix}_{pdf_name}"
        return pdf_name

    def _param_name(self, name: str) -> str:
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name

    def _resolve_param_spec(self, name: str, spec: tuple):
        override = self.kwargs.pop(name, None)
        if override is None:
            if len(spec) == 3:
                title = self._param_name(name)
                init, min_val, max_val = spec
            elif len(spec) == 4:
                title, init, min_val, max_val = spec
            else:
                raise ValueError(
                    f"Invalid par_specs entry for {name}. Expected (init, min, max) or (title, init, min, max)."
                )
            return title, init, min_val, max_val

        if isinstance(override, (tuple, list)):
            if len(override) == 3:
                title = self._param_name(name)
                init, min_val, max_val = override
            elif len(override) == 4:
                title, init, min_val, max_val = override
            else:
                raise ValueError(
                    f"Override for {name} must have length 3 or 4 when tuple/list is provided."
                )
        elif isinstance(override, (int, float)):
            if len(spec) == 3:
                title = self._param_name(name)
                _, min_val, max_val = spec
            elif len(spec) == 4:
                title, _, min_val, max_val = spec
            else:
                raise ValueError(
                    f"Invalid par_specs entry for {name}. Expected (init, min, max) or (title, init, min, max)."
                )
            init = float(override)
        else:
            raise ValueError(f"Unsupported override type for {name}: {type(override)}")

        return title, init, min_val, max_val

    def init_params(self, par_specs: Optional[dict] = None):
        if par_specs is not None:
            self.par_specs = par_specs

        self._parameter_values = {}
        for name, spec in self.par_specs.items():
            title, init, min_val, max_val = self._resolve_param_spec(name, spec)
            full_name = self._param_name(name)
            self._parameter_values[name] = ROOT.RooRealVar(full_name, title, init, min_val, max_val)

        return self._parameter_values

    def _params(self):
        return list(self._parameter_values.values())

    def params(self, floating_only=True):
        """
        Return model parameters, optionally excluding constant parameters.
        """
        params = []
        for par in self._params():
            if floating_only and par.isConstant():
                continue
            params.append(par)
        return params

    def get_param(self, name: str):
        """
        Get a parameter by name.
        """
        for par in self.params(floating_only=False):
            if par.GetName() == name:
                return par
        raise ValueError(f"Parameter {name} not found in model parameters")
    
    def set_param(self, name: str, value: float, constant=False):
        p = self.get_param(name)
        p.setVal(value)
        if constant:
            p.setConstant(True)
        else:
            p.setConstant(False)
    
    def set_params(self, name_value_dict: dict, constant=False):
        """
        Set multiple parameters from a dictionary.
        """
        if name_value_dict is None:
            return
        for name, value in name_value_dict.items():
            self.set_param(name, value, constant=constant)
    
    def randomize_params(
            self,
            custom_ranges: Optional[dict] = None,
            rng: Optional[np.random.Generator] = None,
    ):
        """
        Randomize the parameters of the model within their ranges.
        """

        custom_ranges = {} if custom_ranges is None else custom_ranges
        rng = np.random.default_rng() if rng is None else rng

        for par in self.params():
            if par.isConstant():
                continue

            if par.GetName() in custom_ranges:
                min_val, max_val = custom_ranges[par.GetName()]
            else:
                min_val, max_val = par.getMin(), par.getMax()
            
            random_val = rng.uniform(min_val, max_val)
            par.setVal(random_val)

    def print(self):
        """
        Print the model parameters.
        """
        print(f"Model: {self.name}")
        for par in self.params():
            print_par(par)

def evaluate_pdf(
        x: ROOT.RooRealVar,
        model: RooFitModel,
        x_vals,
    ):
    values = np.zeros_like(x_vals)
    normalization_set = ROOT.RooArgSet(x)
    x.setVal(x_vals[0])
    first = model.pdf.getVal(normalization_set)
    for i, xv in enumerate(x_vals):
        x.setVal(xv)
        values[i] = model.pdf.getVal(normalization_set)
    norm = values[0] / first
    return values / norm


# RooFit model precursor
class ModelPrimitive:
    """Declarative model primitive: model callable + args + kwargs.
    This is a simple wrapper that allows us to store the information needed to instantiate a model without actually doing so until necessary.
    """

    def __init__(
        self,
        model_cls: type[RooFitModel],
        *args: Any,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_cls = model_cls
        self.args = args
        self.kwargs = kwargs
        self.name = name if name is not None else getattr(model_cls, "name", None)

    def __call__(self, x: Any) -> Any:
        return self.model_cls(x, *self.args, **self.kwargs)


# Mixture model helper functions
def stick_breaking_weights(n_components: int, prefix: str = "", **custom_ranges):
    """
    Generate stick-breaking weights for a mixture model.
    """
    stick_proportions = [
        ROOT.RooRealVar(
            f"{prefix}raw_weight_{i}",
            f"Stick proportion {i}",
            1/(n_components - i),
            *custom_ranges.get(f"raw_weight_{i}", (0, 1))
        ) for i in range(n_components - 1)
    ]
    stick_proportions.append(ROOT.RooRealVar(
        f"{prefix}raw_weight_{n_components-1}",
        f"Stick proportion {n_components-1}",
        1 # Fixed to 1
    ))

    weights = []
    for i in range(n_components):
        prod_terms = [f"(1-{prefix}raw_weight_{j})" for j in range(i)]
        prod_str = "*".join(prod_terms) if prod_terms else "1"
        weight = ROOT.RooFormulaVar(
                f"{prefix}weight_{i}",
                    f"Weight for component {i}",
                f"{prefix}raw_weight_{i} * {prod_str}",
                    ROOT.RooArgList(*stick_proportions[:i+1])
        )
        weights.append(weight)
    
    return weights, stick_proportions[:-1]

def softmax_weights(n_components: int, prefix: str = "", **custom_ranges):
    """
    Generate normalization construction weights for a mixture model.
    """
    unnormalized_weights = []
    for i in range(n_components):
        w = ROOT.RooRealVar(
            f"{prefix}raw_weight_{i}",
                f"Unnormalized weight for component {i}",
                custom_ranges.get(f"raw_weight_{i}", 0+i),
            )
        if i == 0:
            w.setConstant(True)
            # w.setConstant(False) # In general this should be fixed but for now let it float
        else:
            w.setConstant(False)
        unnormalized_weights.append(w)

    sum_weights_str = "+".join(f"exp({w.GetName()})" for w in unnormalized_weights)
    weights = [
        ROOT.RooFormulaVar(
            f"{prefix}weight_{i}",
            f"Weight for component {i}",
            f"exp({unnormalized_weights[i].GetName()})/({sum_weights_str})",
            ROOT.RooArgList(*unnormalized_weights)
        ) for i in range(n_components)
    ]
    return weights, unnormalized_weights

def mixture_pdf(
    weights: Sequence[ROOT.RooAbsReal],
    pdfs: Sequence[ROOT.RooAbsPdf],
    name: str = "pdf",
    ):
    assert len(weights) == len(pdfs), "Weights and PDFs must have the same length"

    pdf = ROOT.RooAddPdf(
        name,
        "Mixture PDF",
        ROOT.RooArgList(*pdfs),
        ROOT.RooArgList(*weights[:-1]),
    )
    return pdf

def extended_weights(
        n_components: int,
        n_bkg: int,
    prefix: str = "",
    ):

    raw_weights = [
        ROOT.RooRealVar(
            f"{prefix}raw_weight_{i}",
            f"Raw Weight for component {i}",
            1/n_components, 0, 2*n_bkg
        ) for i in range(n_components)
    ]
    weights = [
        ROOT.RooFormulaVar(
            f"{prefix}weight_{i}",
            f"Weight for component {i}",
            f"{n_bkg}*raw_weight_{i}",
            ROOT.RooArgList(raw_weights[i])
        ) for i in range(n_components)
    ]
    return weights, raw_weights

def extended_mixture_pdf(
    weights: Sequence[ROOT.RooAbsReal],
    pdfs: Sequence[ROOT.RooAbsPdf],
    name: str = "pdf"
):
    assert len(weights) == len(pdfs), "Weights and PDFs must have the same length"

    pdf = ROOT.RooAddPdf(
        name,
        "Extended Mixture PDF",
        ROOT.RooArgList(*pdfs),
        ROOT.RooArgList(*weights),
    )
    return pdf

# Mixture models
class MixtureModel(RooFitModel):

    def initialize(self, x, n_components: int, **kwargs):
        self.n_components = n_components
        self.name = f"Mixture-{n_components}"

        self.init_weights()

        self.pdfs: Sequence[ROOT.RooAbsPdf] = []
        self.init_pdfs(x)

        if kwargs.get("extended", False):
            self.pdf = extended_mixture_pdf(
                self.weights,
                self.pdfs,
                name=self._pdf_name(self.name),
            )
        else:
            self.pdf = mixture_pdf(
                self.weights,
                self.pdfs,
                name=self._pdf_name(self.name),
            )
    
    def init_weights(self, **custom_ranges):
        if self.kwargs.get("stick_breaking", False):
            self.weights, self.raw_weights = stick_breaking_weights(
                self.n_components, prefix=f"{self.prefix}_", **custom_ranges
            )
        elif self.kwargs.get("extended", False):
            n_data = self.kwargs.get("n_data", 0)
            if n_data == 0:
                raise ValueError("n_data must be provided for extended mixture models")
            self.weights, self.raw_weights = extended_weights(
                self.n_components, n_data, prefix=f"{self.prefix}_"
            )
        else:
            self.weights, self.raw_weights = softmax_weights(
                self.n_components, prefix=f"{self.prefix}_"
            )
    
    def init_pdfs(self, x):
        """
        Initialize the PDFs for the mixture model.
        The PDF names should be in the format "pdf_{i}" where i is the index of the component.
        """
        raise NotImplementedError("Subclasses must implement init_pdfs")

class ExponentialMixtureModel(MixtureModel):
    rate_max = 50

    default_par_specs = {}

    def randomize_params(
            self,
            custom_ranges: Optional[dict] = None,
            rng: Optional[np.random.Generator] = None,
    ):
        custom_ranges = {} if custom_ranges is None else custom_ranges
        rng = np.random.default_rng() if rng is None else rng
        if not custom_ranges:

            # Rates
            if self.kwargs.get("random_rates_2", False):
                rate_diffs = rng.uniform(0.15, 1, size=self.n_components)
                rates = np.cumsum(rate_diffs)
            else:
                rates = []
                rate = 0
                for i in range(self.n_components):
                    rate += rng.uniform(0.01, 3)
                    if i == len(self.raw_rates) - 1:
                        rate += rng.uniform(0, 20)
                    rates.append(rate)
            
            for raw_rate, rate in zip(self.raw_rates, rates):
                raw_rate.setVal(rate)

            # Weights
            weights = rng.dirichlet(np.ones(self.n_components))
            if self.kwargs.get("extended", False):
                pass # Already in the correct form
            elif self.kwargs.get("stick_breaking", False):
                raise NotImplementedError("Randomization for stick-breaking weights is not implemented.")
            else: # Softmax is default
                weights = np.log(weights / weights[0])

            for raw_weight, weight in zip(self.raw_weights, weights):
                raw_weight.setVal(weight)

        else:
            super().randomize_params(custom_ranges=custom_ranges)

    def init_rates(self, x):
        self.name = f"Exponential{self.name}"
        assert "data_mean" in self.kwargs, "data_mean must be provided to initialize rates"
        data_mean = self.kwargs["data_mean"]

        rate_scaling = -1/(data_mean - x.getMin())
        
        raw_rate_specs = [(0.1+i, 0, self.rate_max) for i in range(self.n_components)]

        self.par_specs = {
            f"raw_rate_{i}": (f"Raw rate for exponential {i}", *raw_rate_specs[i])
            for i in range(self.n_components)
        }
        self.init_params(self.par_specs)
        self.raw_rates = [self._parameter_values[f"raw_rate_{i}"] for i in range(self.n_components)]

        self.rates = [
            ROOT.RooFormulaVar(
                self._param_name(f"rate_{i}"),
                f"Rate scaled by data for exponential {i}",
                f"{rate_scaling}*{raw_rate.GetName()}",
                ROOT.RooArgList(self.raw_rates[i])
            ) for i, raw_rate in enumerate(self.raw_rates)
        ]
    
    def init_pdfs(self, x):
        self.init_rates(x)

        if "max_tail_prob" in self.kwargs:    
            self.weights[0].setRange(0, self.kwargs["max_tail_prob"])

        # print("Using MY Exponential PDFs for the mixture model.")
        self.pdfs = [
            ROOT.RooExponential(
            # ROOT.MyExponential(
                self._param_name(f"pdf_{i}"),
                f"Exponential PDF {i}",
                x, self.rates[i]
            ) for i in range(self.n_components)    
        ]
    
    def _params(self):
        """
        Return the parameters of the model.
        """
        return self.raw_rates + self.raw_weights

class ExponentialMixtureModel_Ordered(ExponentialMixtureModel):
    """Exponential mixture with ordered rates."""
    rate_min = 0.1
    rate_diff_min = 0.05

    def randomize_params(
            self,
            custom_ranges: Optional[dict] = None,
            rng: Optional[np.random.Generator] = None,
    ):
        custom_ranges = {} if custom_ranges is None else custom_ranges
        rng = np.random.default_rng() if rng is None else rng
        if not custom_ranges:
            for i, raw_rate_diff in enumerate(self.raw_rate_diffs):
                if i == 0:
                    rate_diff = rng.uniform(self.rate_min, 1)
                elif i < len(self.raw_rate_diffs)-1:
                    rate_diff = rng.uniform(self.rate_diff_min, 1)
                else:
                    rate_diff = rng.uniform(self.rate_diff_min, 100)
                
                raw_rate_diff.setVal(rate_diff)
            for raw_weight in self.raw_weights:
                if not raw_weight.isConstant():
                    raw_weight.setVal(0)
        else:
            return super().randomize_params(custom_ranges, rng)

    def init_rates(self, x, random_initialization=False):
        self.name = f"ExponentialMixture-Ordered-{self.n_components}"
        assert "data_mean" in self.kwargs, "data_mean must be provided to initialize rates"
        rate_scaling = -1/(self.kwargs["data_mean"] - x.getMin())

        if "random_initialization" in self.kwargs:
            random_initialization = self.kwargs["random_initialization"]
        elif "random" in self.kwargs:
            random_initialization = self.kwargs["random"]

        mult = 1.0
        if random_initialization:
            mult = np.random.uniform(0.5, 2.0)

        self.par_specs = {
            f"raw_rate_diff_{i}": (
                f"Inverse rate difference {i}",
                0.5*mult,
                self.rate_min if i == 0 else self.rate_diff_min,
                100,
            )
            for i in range(self.n_components)
        }
        self.init_params(self.par_specs)
        self.raw_rate_diffs = [self._parameter_values[f"raw_rate_diff_{i}"] for i in range(self.n_components)]

        self.rates = [
            ROOT.RooFormulaVar(
                f"rate_{i}",
                f"Ordered Rate for exponential {i}",
                f"{rate_scaling}*({'+'.join([rd.GetName() for rd in self.raw_rate_diffs[:i+1]])})",
                ROOT.RooArgList(*self.raw_rate_diffs[:i+1])
            ) for i in range(self.n_components)
        ]

    def _params(self):
        """
        Return the parameters of the model.
        """
        return self.raw_rate_diffs + self.raw_weights

class LomaxMixtureModel(MixtureModel):
    """Exponential mixture with finite Gamma prior on the rates, which results in a finite mixture of Lomax distributions."""
    name = "LomaxMixtureModel"
    def init_alphas(self, x):
        self.alpha_specs = {
            f"alpha_{i}": (f"Shape parameter {i}", *self.kwargs.get(f"alpha_{i}", (2+i, 0.01, 10)))
            for i in range(self.n_components)
        }
    
    def init_betas(self, x):
        self.beta_specs = {
            f"beta_{i}": (f"Scale parameter {i}", *self.kwargs.get(f"beta_{i}", (0.005, 0.000001, 1)))
            for i in range(self.n_components)
        }

    def init_pdfs(self, x):
        self.init_alphas(x)
        self.init_betas(x)
        self.par_specs = {**self.alpha_specs, **self.beta_specs}
        self.init_params(self.par_specs)
        self.alphas = [self._parameter_values[f"alpha_{i}"] for i in range(self.n_components)]
        self.betas = [self._parameter_values[f"beta_{i}"] for i in range(self.n_components)]
        
        self.pdfs = [
            ROOT.RooGenericPdf(
                f"pdf_{i}", f"Lomax {i}",
                f"(alpha_{i}*beta_{i}) * (1 + x*beta_{i})^(-alpha_{i}-1)",
                ROOT.RooArgList(
                    x,
                    self.betas[i],
                    self.alphas[i],
                )
            ) for i in range(self.n_components)
        ]

    def _params(self):
        return self.alphas + self.betas + self.raw_weights

class GPDMixtureModel(MixtureModel):
    def init_pdfs(self, x):
        self.init_powers()
        if "max_x" in self.kwargs:
            self.x_max = self.kwargs["max_x"]
        else:
            print(f"Warning: max_x not provided for GPDMixtureModel, using x.getMax() = {x.getMax()}")
            self.x_max = x.getMax()
        
        self.pdfs = [
            ROOT.RooGenericPdf(
                f"pdf_{i}",
                f"{self.weights[i].GetName()}*pow(1 - ({x.GetName()}-{x.getMin()})/({self.x_max}-{x.getMin()}), {self.powers[i].GetName()})",
                ROOT.RooArgList(self.weights[i], self.powers[i], x)
            ) for i in range(self.n_components)
        ]

    def init_powers(self):
        self.par_specs = {
            f"power_{i}": (f"Power for component {i}", 5 + i*10, -1, 100_000)
            for i in range(self.n_components)
        }
        self.init_params(self.par_specs)
        self.powers = [self._parameter_values[f"power_{i}"] for i in range(self.n_components)]
    
    def _params(self):
        """
        Return the parameters of the model.
        """
        return self.powers + self.raw_weights


# Alternative models
class GeneralizedPareto(RooFitModel):
    name = "GPD"
    formula = "$(N/p_1)*(1 + p_2*(x-x_min)/p_1)^{-1-1/p_2}$"
    default_par_specs = {
        "p1": (0.5, 0, 10),
        "p2": (1000, 0.01, 10000),
    }

    def initialize(self, x, **kwargs):
        p = self._parameter_values

        self.pdf = ROOT.RooGenericPdf(
            self._pdf_name(self.name),
            f"(1/{p['p2'].GetName()})*pow(1 + {p['p1'].GetName()}*({x.GetName()}-{x.getMin()})/{p['p2'].GetName()}, -1 - 1/{p['p1'].GetName()})",  # Formula for the PDF
            ROOT.RooArgList(p["p2"], p["p1"], x)  # Arguments for the formula
        )

class Dijet(RooFitModel):
    name="Dijet"
    default_par_specs = {
        "p1": (-20, -100, 100),
        "p2": (-50, -100, 100),
        "p3": (-10, -100, 100),
    }

    def initialize(self, x, **kwargs):
        p = self._parameter_values

        fn = lambda x, p1, p2, p3: f"pow(1 - {x}, {p1}) * pow({x}, {p2} + {p3} * TMath::Log({x}))"
        self.pdf = ROOT.RooGenericPdf(
            self._pdf_name(self.name),
            fn(f"{x.GetName()}/13000", p["p1"].GetName(), p["p2"].GetName(), p["p3"].GetName()),  # Formula for the PDF
            ROOT.RooArgList(x, p["p1"], p["p2"], p["p3"])  # Arguments for the formula
        )

class f1(RooFitModel):
    name="f_1"
    formula = "$N x^{p_1 + p_2 \\log(x)}$"
    default_par_specs = {
        "p1": (5.7, 1, 11),
        "p2": (-0.78, -1.0, -0.5),
    }

    def initialize(self, x, **kwargs):
        p = self._parameter_values

        self.pdf = ROOT.RooGenericPdf(
            self._pdf_name(self.name),
            f"pow(x,{p['p1'].GetName()}+{p['p2'].GetName()}*TMath::Log(x))",  # Formula for the PDF
            ROOT.RooArgList(p["p1"], p["p2"], x),  # Arguments for the formula
        )

class f2(RooFitModel):
    name= "f_2"
    formula = "$N\\exp(p_1*x)*x^{-p_2^2}$"
    default_par_specs = {
        "p1": (-0.0016, -0.1, 0),
        "p2": (1.8, 1, 3),
    }

    def initialize(self, x, **kwargs):
        p = self._parameter_values

        self.pdf = ROOT.RooGenericPdf(
            self._pdf_name(self.name),
            f"7e+10*exp({p['p1'].GetName()}*x)*pow(x,-1*{p['p2'].GetName()}*{p['p2'].GetName()})",  # Formula for the PDF
            ROOT.RooArgList(p["p1"], p["p2"], x)  # Arguments for the formula
        )

class f3(RooFitModel):
    name= "f_3"
    formula = "$N(1 + p_1 x)^{-p_2^2}$"
    default_par_specs = {
        "p1": (0.00228, 0, 0.05),
        "p2": (2.7013689, 2, 4),
    }

    def initialize(self, x, **kwargs):
        p = self._parameter_values

        self.pdf = ROOT.RooGenericPdf(
            self._pdf_name(self.name),
            f"8760*pow(1+{p['p1'].GetName()}*x, -1*{p['p2'].GetName()}*{p['p2'].GetName()})",  # Formula for the PDF
            ROOT.RooArgList(p["p1"], p["p2"], x)  # Arguments for the formula
        )

class f4(RooFitModel):
    name = "f_4"
    formula = "$N(1 + p_1 x)^{-p_2 - p_3 x}$"
    default_par_specs = {
        "p1": (0.029456453, 0, 0.1),
        "p2": (3.8645171, 1, 9),
        "p3": (0.00027, 0, 0.01),
    }

    def initialize(self, x, **kwargs):
        p = self._parameter_values

        self.pdf = ROOT.RooGenericPdf(
            self._pdf_name(self.name),
            f"2124447*pow(1+{p['p1'].GetName()}*x, -{p['p2'].GetName()} - {p['p3'].GetName()}*x)",  # Formula for the PDF
            ROOT.RooArgList(p["p1"], p["p2"], p["p3"], x)  # Arguments for the formula
        )


# Models with signal
def signal_lower_bound(x, signal, background, *,
                       signal_range, nb=None,
                       ngrid=1001, margin=0.01):
    """Approximate bound at the current component parameters."""
    obs = ROOT.RooArgSet(x)
    old_x = x.getVal()
    bounds = []

    lo = max(x.getMin(), signal_range[0])
    hi = min(x.getMax(), signal_range[1])
    if lo >= hi:
        raise ValueError("Signal region does not overlap the fit range")

    try:
        for value in np.linspace(lo, hi, ngrid):
            x.setVal(float(value))
            s = signal.getVal(obs)
            b = background.getVal(obs)

            if not np.isfinite(s + b) or s < 0 or b < 0:
                raise ValueError("Invalid component PDF value")

            if nb is None:
                if s > b:
                    bounds.append(-b / (s - b))
            elif s > 0:
                bounds.append(-nb * b / s)
    finally:
        x.setVal(old_x)

    if not bounds:
        raise ValueError("No finite lower bound found in signal region")

    return (1.0 - margin) * max(bounds)

class SignalPlusBackgroundModel(RooFitModel):
    name = "SignalPlusBackgroundModel"
    par_specs = {}

    def initialize(self, signal_model: RooFitModel, background_model: RooFitModel, **kwargs):
        self.signal_model = signal_model
        self.background_model = background_model

        n_obs = kwargs.get("n_obs", 5036)

        # Get bounds on the signal fraction
        signal_range = kwargs.get("signal_range", (
            signal_model.get_param("mean").getVal() - 2 * signal_model.get_param("sigma").getVal(),
            signal_model.get_param("mean").getVal() + 2 * signal_model.get_param("sigma").getVal(),
        ))
        signal_lower_bound_value = signal_lower_bound(
            x=kwargs.get("x"),
            signal=signal_model.pdf,
            background=background_model.pdf,
            signal_range=signal_range,
            nb=n_obs,
        )

        min_sig = kwargs.get("min_sig", signal_lower_bound_value)
        max_sig = kwargs.get("max_sig", 100)

        self.par_specs = {
            "n_sig": ("Number of signal events", 0, min_sig, max_sig),
        }

        self.init_params(self.par_specs)
        p = self._parameter_values

        self.sig_frac = ROOT.RooFormulaVar(
            "sig_frac",
            "Signal fraction",
            f"n_sig / {n_obs}",
            ROOT.RooArgList(p["n_sig"])
        )
        self.bkg_frac = ROOT.RooFormulaVar(
            "bkg_frac",
            "Background fraction",
            "1 - sig_frac",
            ROOT.RooArgList(self.sig_frac)
        )

        self.pdf = ROOT.RooRealSumPdf(
            "signal_plus_background_pdf",
            "Signal plus Background PDF",
            ROOT.RooArgList(self.signal_model.pdf, self.background_model.pdf),
            ROOT.RooArgList(self.sig_frac, self.bkg_frac)
        )

    def _params(self):
        p = self._parameter_values
        return [p["n_sig"]] + self.signal_model.params() + self.background_model.params()


class SignalPlusBackgroundModelExtended(RooFitModel):
    name = "SignalPlusBackgroundModelExtended"
    par_specs = {}

    def initialize(self, signal_model: RooFitModel, background_model: RooFitModel, **kwargs):
        self.signal_model = signal_model
        self.background_model = background_model

        n_obs = kwargs.get("n_obs", 5036)

        # Get bounds on the signal fraction
        signal_range = kwargs.get("signal_range", (
            signal_model.get_param("mean").getVal() - 2 * signal_model.get_param("sigma").getVal(),
            signal_model.get_param("mean").getVal() + 2 * signal_model.get_param("sigma").getVal(),
        ))
        signal_lower_bound_value = signal_lower_bound(
            x=kwargs.get("x"),
            signal=signal_model.pdf,
            background=background_model.pdf,
            signal_range=signal_range,
            nb=n_obs,
        )

        min_sig = kwargs.get("min_sig", signal_lower_bound_value)
        max_sig = kwargs.get("max_sig", 100)
        if min_sig > max_sig:
            raise ValueError("min_sig must not exceed max_sig")

        self.par_specs = {
            "n_sig": ("Number of signal events", 0, min_sig, max_sig),
            "n_bkg": ("Number of background events", n_obs, 0.5*n_obs, 2*n_obs),
        }
        self.init_params(self.par_specs)
        p = self._parameter_values

        self.pdf = ROOT.RooAddPdf(
            "signal_plus_background_pdf",
            "Signal plus Background PDF",
            ROOT.RooArgList(self.signal_model.pdf, self.background_model.pdf),
            ROOT.RooArgList(p["n_sig"], p["n_bkg"])
        )

    
    def _params(self):
        p = self._parameter_values
        return [p["n_sig"], p["n_bkg"]] + self.signal_model.params() + self.background_model.params()

class GaussianSignalModel(RooFitModel):
    name = "Gaussian"
    par_specs = {}

    def initialize(self, x, mean, width, **kwargs):
        self.par_specs = {
            "mean": ("Signal mean", mean, 0.9*mean, 1.1*mean),
            "sigma": ("Signal sigma", width, 0.8*width, 1.2*width),
        }
        self.init_params(self.par_specs)
        p = self._parameter_values

        self.pdf = ROOT.RooGaussian(
            self._param_name("sig_pdf"),
            "Gaussian Signal PDF",
            x,
            p["mean"],
            p["sigma"],
        )
        p["mean"].setConstant(True)
        p["sigma"].setConstant(True)