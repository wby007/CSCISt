from intake import imports

from isegm.data.compose import ComposeDataset, ProportionalComposeDataset
from .berkeley import BerkeleyDataset
from .davis import DavisDataset
from .grabcut import GrabCutDataset
from .sbd import SBDDataset, SBDEvaluationDataset
from .pascalpart import PascalPartDataset, PascalPartEvaluationDataset
from .pascalpart_fmg import PascalPartFMGDataset
from .pascalpart_mmt_fmg import PascalPartMMTFMGDataset
from .partimagenet import PartINEvaluationDataset
from .sa1b import SA1BDataset
from .pascalvoc import PascalVocDataset
from .brats import BraTSDataset
from .ssTEM import ssTEMDataset
from .oai_zib import OAIZIBDataset
from .COCO2014 import COCO2014Dataset
from .COCO2017 import COCO2017Dataset
from .coco_lvis import CocoLvisDataset
from .COCOMVal import COCOMValDataset
