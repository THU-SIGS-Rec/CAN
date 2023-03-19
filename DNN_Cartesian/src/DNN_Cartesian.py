import torch
from copy import deepcopy
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from collections import OrderedDict


class DNN_Cartesian(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DNN_Cartesian", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DNN_Cartesian, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        
        self.feature_map = deepcopy(self.feature_map)
        self.feature_map.set_column_index()
        categorical_field = []
        for feature, feature_spec in self.feature_map.features.items():
            if feature_spec["type"] == "categorical":
                categorical_field.append((feature, feature_spec))
        for i in range(len(categorical_field)):
            for j in range(i+1, len(categorical_field)):
                spec_a = categorical_field[i][1]
                spec_b = categorical_field[j][1]
                if spec_a["vocab_size"] > 1000 and spec_b["vocab_size"] > 1000:
                    continue
                spec_new = {"type": "categorical",
                            "vocab_size": spec_a["vocab_size"] * spec_b["vocab_size"],
                            "oov_idx": spec_a["vocab_size"] * spec_b["vocab_size"] - 1,
                            "padding_idx": 0,
                            "source": "combine"
                            }
                self.feature_map.features["{}_&_{}".format(categorical_field[i][0], categorical_field[j][0])] = spec_new
        
        self.embedding_layer = FeatureEmbedding(self.feature_map, embedding_dim)
        self.mlp = MLP_Block(input_dim=self.feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def get_inputs(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
            if spec["source"] == "combine":
                source_features = feature.split('_&_')
                idx = [self.feature_map.get_column_index(source) for source in source_features]
                vocabsize = [self.feature_map.features[source]['vocab_size'] for source in source_features]
                X_dict[feature] = inputs[:, idx[0]]
                for i in range(1, len(source_features)):
                    X_dict[feature] = X_dict[feature] * vocabsize[i] + inputs[:, idx[i]]
                X_dict[feature] = X_dict[feature].to(self.device)
            else:
                X_dict[feature] = inputs[:, self.feature_map.get_column_index(feature)].to(self.device)
        return X_dict
    
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, dynamic_emb_dim=True)
        y_pred = self.mlp(feature_emb.flatten(start_dim=1))
        return_dict = {"y_pred": y_pred}
        return return_dict
        
