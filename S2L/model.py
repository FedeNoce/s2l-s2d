import torch.nn as nn
from S2L.wav2vec import Wav2Vec2Model

class Speech2Land(nn.Module):
    def __init__(self, args):
        super(Speech2Land, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, L*3)
        landmarks: (batch_size, seq_len, L*3)
        """
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.hidden_size = args.feature_dim
        self.drop_prob = 0.2
        self.num_layers = args.num_layers

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.output_size = args.landmarks_dim

        self.lstm = nn.LSTM(input_size=args.audio_feature_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=self.drop_prob)


        self.output_mapper = nn.Linear(self.hidden_size*2, self.output_size)
        nn.init.constant_(self.output_mapper.weight, 0)
        nn.init.constant_(self.output_mapper.bias, 0)


    def forward(self, audio, displacements, criterion):

        frame_num = displacements.shape[1]

        hidden_states = self.audio_encoder(audio, frame_num=frame_num).last_hidden_state

        displacements_emb, _ = self.lstm(hidden_states)

        displacements_pred = self.output_mapper(displacements_emb)

        loss = criterion(displacements_pred, displacements)

        return loss

    def predict(self, audio, template):

        hidden_states = self.audio_encoder(audio).last_hidden_state

        shift_emb, _ = self.lstm(hidden_states)

        landmarks_pred = self.output_mapper(shift_emb) + template.unsqueeze(0)

        return landmarks_pred
