from fastapi import FastAPI
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers    
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
        
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):        
        # x => (64, 178, 1)
        
        # pass data through the LSTM
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        
        # we take the final timestep's output for classification
        final_out = lstm_out[:, -1, :]
        
        # apply dropout
        x = self.dropout(final_out)
        
        # classify 
        x = self.fc(x)
        
        return x
    

app = FastAPI()


@app.get("/predict_EEG")
async def root():
    # sampling rate is 178hz
    ex_signal = "135,190,229,223,192,125,55,-9,-33,-38,-10,35,64,113,152,164,127,50,-47,-121,-138,-125,-101,-50,11,39,24,48,64,46,13,-19,-61,-96,-130,-132,-116,-115,-71,-14,25,19,6,9,21,13,-37,-58,-33,5,47,80,101,88,73,69,41,-13,-31,-61,-80,-77,-66,-43,5,87,129,121,88,12,-76,-150,-207,-186,-165,-148,-103,-33,40,94,75,8,-81,-155,-227,-262,-233,-218,-187,-126,-65,-12,27,61,49,9,-46,-124,-210,-281,-265,-181,-89,-4,53,53,38,43,31,34,9,-7,-34,-70,-84,-101,-70,-11,42,62,66,74,64,59,56,36,-11,-30,-43,-23,8,42,77,103,135,121,79,59,43,54,90,111,107,64,32,18,-25,-69,-65,-44,-33,-57,-88,-114,-130,-114,-83,-53,-79,-72,-85,-109,-98,-72,-65,-63,-11,10,8,-17,-15,-31,-77,-103,-127,-116,-83,-51".split(",")
    
    ex_signal = [float(x) for x in ex_signal]
    
    model = LSTM(1, 64, 2)
    model.load_state_dict(torch.load('weights/best_model.pth'))
    model.eval()

    input_tensor = torch.tensor(ex_signal, dtype=torch.float32).view(1, 178, 1)
    print(input_tensor)
    
    with torch.no_grad():
        probability = torch.sigmoid(model(input_tensor)).item()
        return {"prediction": probability}

