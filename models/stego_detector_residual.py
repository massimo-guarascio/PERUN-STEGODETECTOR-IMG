import torch
import torch.nn as nn


import torch
import torch.nn as nn


class StegoDetectorResidualTorch(nn.Module):
 

    def __init__(self, input_dim, dl_size=128, drop_pcg=0.025):
        super().__init__()

        # Blocco iniziale normale: input_dim -> dl_size
        self.block_0 = nn.Sequential(
            nn.Linear(input_dim, dl_size),
            nn.ReLU(),
            nn.BatchNorm1d(dl_size),
            nn.Dropout(drop_pcg),
        )

        # F_1(x)
        self.block_1 = nn.Sequential(
            nn.Linear(dl_size, dl_size),
            nn.ReLU(),
            nn.BatchNorm1d(dl_size),
            nn.Dropout(drop_pcg),
        )

        # Residual block applicato a x + F_1(x)
        self.residual_block_1 = nn.Sequential(
            nn.Linear(dl_size, dl_size),
            nn.ReLU(),
            nn.BatchNorm1d(dl_size),
            nn.Dropout(drop_pcg),
        )

        # F_2(x)
        self.block_2 = nn.Sequential(
            nn.Linear(dl_size, dl_size),
            nn.ReLU(),
            nn.BatchNorm1d(dl_size),
            nn.Dropout(drop_pcg),
        )

        # Residual block applicato a x + F_2(x)
        self.residual_block_2 = nn.Sequential(
            nn.Linear(dl_size, dl_size),
            nn.ReLU(),
            nn.BatchNorm1d(dl_size),
            nn.Dropout(drop_pcg),
        )

        self.output_layer = nn.Linear(dl_size, 1)

    def forward(self, x):
        # Blocco 0 normale
        x = self.block_0(x)

        # Primo stage residuale
        y = self.block_1(x)             # y = F_1(x)
        z = x + y                       # z = x + F_1(x)
        x = self.residual_block_1(z)    # x = R_1(z)

        # Secondo stage residuale
        y = self.block_2(x)             # y = F_2(x)
        z = x + y                       # z = x + F_2(x)
        x = self.residual_block_2(z)    # x = R_2(z)

        # Output finale
        x = self.output_layer(x)

        return x



class StegoDetectorResidualTorch_new(nn.Module):

    def __init__(self, input_dim, dl_size=128, drop_pcg=0.025):
            super().__init__()
    
            # Blocco iniziale normale: input_dim -> dl_size
            self.block_0 = nn.Sequential(
                nn.Linear(input_dim, dl_size),
                nn.ReLU(),
                nn.BatchNorm1d(dl_size),
                nn.Dropout(drop_pcg),
            )
    
            # F_1(x)
            self.block_1 = nn.Sequential(
                nn.Linear(dl_size, dl_size),
                nn.ReLU(),
                nn.BatchNorm1d(dl_size),
                nn.Dropout(drop_pcg),
            )
    
            # Residual block applicato a x + F_1(x)
            self.residual_block_1 = nn.Sequential(
                nn.Linear(dl_size, dl_size),
                nn.ReLU(),
                nn.BatchNorm1d(dl_size),
                nn.Dropout(drop_pcg),
            )
    
            # Nuovo layer lineare da 128 neuroni
            self.linear_concat = nn.Sequential(
                nn.Linear(dl_size, 128),
                nn.ReLU(),
            )
    
            # Output su vettore concatenato:
            # 128 (block_0) + 128 (linear_concat)
            #self.output_layer = nn.Linear(dl_size + 128, 1)
            self.output_layer = nn.Linear(dl_size, 1)
    
    def forward(self, x):
            # Salvo l'output del blocco 0 per la concatenazione finale
            x0 = self.block_0(x)
    
            # Primo stage residuale
            y = self.block_1(x0)            # y = F_1(x0)
            z = x0 + y                      # z = x0 + F_1(x0)
            x1 = self.residual_block_1(z)   # x1 = R_1(z)
    
            # Nuovo layer lineare
            x1 = self.linear_concat(x1)
    
            # Concatenazione con l'output del blocco 0
            x_sum = x0+x1
    
            # Output finale
            out = self.output_layer(x_sum)
    
            return out