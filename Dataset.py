from torch.utils.data import Dataset
import torchaudio
import torch

def load_audio(audio_path, sample_rate):
    audio_data = torchaudio.load(audio_path)
    audio_data = torch.FloatTensor(audio_data)

    return audio_data

class AudioDataset(Dataset):
    def __init__(self, data, _id, root, audio_conf, frame_length, shift_length, train=True):
        self.data = [sample for sample in data if _id in sample]
        self.root = root
        self.train = train

        if not self.train:
            anomaly_data = [data for data in self.data if "anomaly" in data]
            self.anomaly_num = len(anomaly_data)
            self.normal_num = len(self.data) - self.anomaly_num

        self.frame_length = frame_length
        self.shift_length = shift_length

        self.audio_conf = audio_conf
        """
        Audio Config is a dict type
            num_mel_bins (128):     number of mel bins in audio spectrogram
            target_length (1024):   number of frames is formed after the raw audio go through the filter bank, use 1024
            freqm (0):              frequency mask length, default: 0
            timem (0):              time mask length, default: 0
            mixup (0):              mixup with other wave
            dataset (dcase):        the dataset we apply on is dcase dataset
            mode (train or test):   train mode or test mode
            mean (-4.2677393):      use for normalization
            std (4.5689974):        use for normalization
            noise (False):          does the dataset add noise into audio
        """
        
        self.melbins = audio_conf['num_mel_bins']
        self.target_length = audio_conf['target_length']
        self.freqm = audio_conf['freqm']
        self.timem = audio_conf['timem']
        self.mixup = audio_conf['mixup']
        self.dataset = audio_conf['dataset']
        self.mode = audio_conf['mode']
        self.norm_mean = audio_conf['mean']
        self.norm_std = audio_conf['std']
        self.noise = audio_conf['noise']
        self.sample_rate = audio_conf['sample_rate']
    
    def _wav2fbank(self, filename, filename2=None):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        
        
        # # mixup
        # else:
        #     waveform1, sr = torchaudio.load(filename)
        #     waveform2, _ = torchaudio.load(filename2)

        #     waveform1 = waveform1 - waveform1.mean()
        #     waveform2 = waveform2 - waveform2.mean()

        #     if waveform1.shape[1] != waveform2.shape[1]:
        #         if waveform1.shape[1] > waveform2.shape[1]:
        #             # padding
        #             temp_wav = torch.zeros(1, waveform1.shape[1])
        #             temp_wav[0, 0:waveform2.shape[1]] = waveform2
        #             waveform2 = temp_wav
        #         else:
        #             # cutting
        #             waveform2 = waveform2[0, 0:waveform1.shape[1]]

        #     # sample lambda from uniform distribution
        #     #mix_lambda = random.random()
        #     # sample lambda from beta distribtion
        #     mix_lambda = np.random.beta(10, 10)

        #     mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
        #     waveform = mix_waveform - mix_waveform.mean()
        
        """
        sample rate: 16k
        frame length: 50ms (default)
        shift length: 20ms
        """
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.sample_rate, use_energy=False, frame_length=self.frame_length,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.shift_length)
        
        n_frames = fbank.shape[0]

        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]

        if filename2 == None:
            return fbank, 0
        """
        else:
            return fbank, mix_lambda
        """

    def __getitem__(self, index):
        file_path = self.data[index]
        #audio_data = load_audio(f"{file_path}", sample_rate=self.sample_rate)

        
        # # do mix-up for this sample (controlled by the given mixup rate)
        # if random.random() < self.mixup:
        #     datum = self.data[index]
        #     # find another sample to mix, also do balance sampling
        #     # sample the other sample from the multinomial distribution, will make the performance worse
        #     # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
        #     # sample the other sample from the uniform distribution
        #     mix_sample_idx = random.randint(0, len(self.data)-1)
        #     mix_datum = self.data[mix_sample_idx]
        #     # get the mixed fbank
        #     fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
        #     # initialize the label
        #     label_indices = np.zeros(self.label_num)
        #     # add sample 1 labels
        #     for label_str in datum['labels'].split(','):
        #         label_indices[int(self.index_dict[label_str])] += mix_lambda
        #     # add sample 2 labels
        #     for label_str in mix_datum['labels'].split(','):
        #         label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
        #     label_indices = torch.FloatTensor(label_indices)
        # # if not do mixup
        # else:
        #     datum = self.data[index]
        #     label_indices = np.zeros(self.label_num)
        #     fbank, mix_lambda = self._wav2fbank(datum['wav'])
        #     for label_str in datum['labels'].split(','):
        #         label_indices[int(self.index_dict[label_str])] = 1.0

        #     label_indices = torch.FloatTensor(label_indices)
        

        # no mixup
        fbank, _ = self._wav2fbank(file_path)

        
        # # SpecAug, not do for eval set
        # freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        # timem = torchaudio.transforms.TimeMasking(self.timem)
        # fbank = torch.transpose(fbank, 0, 1)
        # # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        # fbank = fbank.unsqueeze(0)
        # if self.freqm != 0:
        #     fbank = freqm(fbank)
        # if self.timem != 0:
        #     fbank = timem(fbank)
        # # squeeze it back, it is just a trick to satisfy new torchaudio version
        # fbank = fbank.squeeze(0)
        # fbank = torch.transpose(fbank, 0, 1)
        

        # # normalize the input for both training and test
        # if not self.skip_norm:
        #     fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # # skip normalization the input if you are trying to get the normalization stats.
        # else:
        #     pass

        # Normalize the audio data
        fbank = (fbank - self.norm_mean) / (self.norm_std*2)

        # if self.noise == True:
        #     fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        #     fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

        if self.train:
            return fbank
        else:
            return fbank, 1 if "anomaly" in file_path else 0
    
    def __len__(self):
        return len(self.data)