from pathlib import Path  
from typing import Annotated  
from pydantic import Field  
from prime_rl.utils.config import LogConfig, ModelConfig
from prime_rl.utils.pydantic_config import BaseSettings  
  
class InvestigatorConfig(BaseSettings):  
    """Configures weight investigation."""  
      
    # Model configuration  
    model: ModelConfig = ModelConfig()  
      
    # Logging configuration    
    log: LogConfig = LogConfig()
      
    # Checkpoint paths for comparison  
    checkpoint_path_1: Path | str
    checkpoint_path_2: Path | str
      
    # Output directory  
    output_dir: Path = Path("outputs")
    generate_charts: bool = False
    merge: bool = False