#MMRDR-XAI: clinically explainable diabetic retinopathy grading on MMRDR colour fundus photographs

GRADES = ("No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR")
LESIONS = ("MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD")
LESION_NAMES = ("Microaneurysm", "Hard exudate", "Intraretinal haemorrhage", "Venous beading / IRMA",
                "Neovascularisation", "Vitreous haemorrhage", "Retinal detachment")
NUM_GRADES = len(GRADES)
NUM_LESIONS = len(LESIONS)

__version__ = "0.1.0"
