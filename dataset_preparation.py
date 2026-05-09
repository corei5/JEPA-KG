import pandas as pd
import json
import os
from typing import List, Dict

class JEPAKGDataProcessor:
    """
    Translates raw enterprise and biological data into 
    Context-Target pairs for JEPA-style training.
    """
    
    @staticmethod
    def process_dataco_supply_chain(file_path: str, limit: int = 500) -> List[Dict]:
        """
        Target: Automotive/Logistics Use Case.
        Maps supply chain events to business outcomes.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DataCo file not found at {file_path}")

        # Use 'latin1' encoding as DataCo CSVs often contain special characters
        df = pd.read_csv(file_path, encoding='latin1')
        processed_data = []

        for _, row in df.head(limit).iterrows():
            # OBSERVATION (Context): The graph state before the outcome
            # We serialize key triples representing the logistics environment
            kg_context = (
                f"<|kg_start|> "
                f"({row['Order City']}, hasCustomer, {row['Customer Id']}) "
                f"({row['Category Name']}, shippedFrom, {row['Market']}) "
                f"({row['Shipping Mode']}, linkedTo, Order_{row['Order Id']}) "
                f"<|kg_end|>"
            )
            
            # FUTURE STATE (Target): The ground truth outcome to align with
            # This represents the 'target embedding' the model must predict
            outcome_target = (
                f"Delivery_Status: {row['Delivery Status']}, "
                f"Late_Risk: {row['Late_delivery_risk']}"
            )
            
            # Formulate the training string with Predictor injection point
            # Predictor tokens act as the 'Latent Dynamics' bottleneck
            full_text = f"{kg_context} <|predictor_1|> Outcome: {outcome_target}"
            
            processed_data.append({"text": full_text})
        
        print(f"Successfully processed {len(processed_data)} DataCo records.")
        return processed_data

    @staticmethod
    def process_hetionet_biological(file_path: str, limit: int = 500) -> List[Dict]:
        """
        Target: Life Sciences Use Case.
        Maps molecular subgraphs to functional latent signals.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Hetionet file not found at {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)
        
        nodes = {n['identifier']: n['name'] for n in data['nodes']}
        edges = data['edges']
        processed_data = []

        # We sample every Nth edge to get a diverse therapeutic spread
        step = max(1, len(edges) // (limit * 2))
        for i in range(0, len(edges), step):
            if len(processed_data) >= limit:
                break
                
            edge = edges[i]
            source_name = nodes.get(edge['source_id'], edge['source_id'])
            target_name = nodes.get(edge['target_id'], edge['target_id'])
            kind = edge['kind']

            # OBSERVATION (Context): The biological relationship triple
            kg_context = f"<|kg_start|> ({source_name}, {kind}, {target_name}) <|kg_end|>"
            
            # FUTURE STATE (Target): The predicted functional signal
            # In a real prototype, this would align with experimental efficacy data
            latent_target = f"Efficacy Signal: High latent overlap in {kind} pathway for {source_name}."
            
            full_text = f"{kg_context} <|predictor_1|> {latent_target}"
            processed_data.append({"text": full_text})
            
        print(f"Successfully processed {len(processed_data)} Hetionet biological edges.")
        return processed_data

# --- Usage Example ---
# processor = JEPAKGDataProcessor()
# dataco_samples = processor.process_dataco_supply_chain("DataCoSmartSupplyChainQueries.csv")
# hetio_samples = processor.process_hetionet_biological("hetionet-v1.0.json")
