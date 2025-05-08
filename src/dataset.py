from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class HangmanDataset(Dataset):
    def __init__(self, words, max_word_length=35, max_guesses=20):
        
        self.words = words
        self.vocab =  [chr(i) for i in range(97, 123)] + ['<SOS>', '<MASK>', '<PAD>']
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        # self.reset_states()
        self.max_word_length = max_word_length
        self.max_guesses = max_guesses
        self.state_list = []

        self.target_oneHot = []
        self.character_pos = []
        for word in self.words:
            pos= dict()
            temp = torch.zeros(26)
            for i in range(len(word)):
                ch=word[i]
                temp[self.stoi[ch]] = 1
                pos.setdefault(ch,[])
                pos[ch].append(i)
            
            self.character_pos.append(pos)    
            self.target_oneHot.append(temp)

        self.init_state()


    def __len__(self):
        return len(self.words)

    def init_state(self):
        print('creating dataset')
        for idx in tqdm(range(len(self.words))):
            # pos = self.character_pos[idx]
            word = self.words[idx]
            target_oneHot = self.target_oneHot[idx]
            encoder_input = [self.stoi['<MASK>']] * len(word)
            guess_history = [self.stoi['<SOS>']]
    
            pad_length = self.max_word_length - len(word)
            if pad_length < 0: #Word is longer than max allowed length
                pad_length = 0 #No extra pad token is required
                padded_encoder_input = encoder_input[:self.max_word_length] #word clipping is necessary
                # encoder_input_mask = torch.ones(self.max_word_length)
                encoder_input_mask = torch.zeros(self.max_word_length).bool()
            else:
                padded_encoder_input = torch.tensor(encoder_input + [self.stoi['<PAD>']] * pad_length)
                # encoder_input_mask = torch.cat((torch.ones(len(word)), torch.zeros(pad_length)), dim=0)
                encoder_input_mask = torch.cat((torch.zeros(len(word)), torch.ones(pad_length)), dim=0).bool()
    
            padded_guess_history = torch.tensor(guess_history + [self.stoi['<PAD>']] * (self.max_guesses -1))
            # guess_history_mask = torch.zeros(self.max_guesses)
            # guess_history_mask[0] = 1
            guess_history_mask = torch.ones(self.max_guesses).bool()
            guess_history_mask[0] = False

    
            self.state_list.append({'padded_encoder_input' : padded_encoder_input,
                    'encoder_input_mask' : encoder_input_mask,
                    'padded_guess_history' : padded_guess_history,
                    'guess_history_mask' : guess_history_mask,
                    'target_oneHot' : target_oneHot,
                    'gussed_characters_freq' : torch.zeros(26),
                    'word_id' : idx})
        

    def __getitem__(self, idx):
        return self.state_list[idx], torch.ones(1), torch.zeros(1), torch.zeros(1)


    def update_state(self, state, pred, guess_number, wordNotYetGuessed, incorrectGuesses, trials): 
        # Clone only the needed tensors
        state = {
            'padded_encoder_input': state['padded_encoder_input'].clone(),
            'padded_guess_history': state['padded_guess_history'].clone(),
            'guess_history_mask': state['guess_history_mask'].clone(),
            'target_oneHot': state['target_oneHot'].clone(),
#             'gussed_characters_freq': state['gussed_characters_freq'],  # not modified
            'encoder_input_mask': state['encoder_input_mask'],          # not modified
            'word_id': state['word_id']
        }

        wordNotYetGuessed = wordNotYetGuessed.clone()
        incorrectGuesses = incorrectGuesses.clone()

        ch_id = torch.argmax(pred, dim=-1)  # (batch_size,)
        ch_list = [self.itos[i.item()] for i in ch_id]

        # Vectorized updates
        trials += wordNotYetGuessed
        state['padded_guess_history'][:, guess_number] = ch_id
        state['guess_history_mask'][:, guess_number] = False

        for b, ch in enumerate(ch_list):
            if wordNotYetGuessed[b] == 1:
                if state['target_oneHot'][b, ch_id[b]] == 1:
                    positions = self.character_pos[state['word_id'][b]][ch]
                    for pos in positions:
                        state['padded_encoder_input'][b, pos] = ch_id[b]
                    state['target_oneHot'][b, ch_id[b]] = 0

                    if state['target_oneHot'][b].sum().item() == 0:
                        wordNotYetGuessed[b] = 0
                else:
                    incorrectGuesses[b] += 1

        return state, wordNotYetGuessed, incorrectGuesses, trials




    
    
    
    
    
    
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class HangmanTestDataset(Dataset):
    def __init__(self, words, max_word_length=35, max_guesses=20):
        
        self.words = words
        self.vocab =  [chr(i) for i in range(97, 123)] + ['<SOS>', '<MASK>', '<PAD>']
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        # self.reset_states()
        self.max_word_length = max_word_length
        self.max_guesses = max_guesses
        self.state_list = []

        self.target_oneHot = []
        self.character_pos = []
        for word in self.words:
            pos= dict()
            temp = torch.zeros(26)
            for i in range(len(word)):
                ch=word[i]
                temp[self.stoi[ch]] = 1
                pos.setdefault(ch,[])
                pos[ch].append(i)
            
            self.character_pos.append(pos)    
            self.target_oneHot.append(temp)

        self.init_state()


    def __len__(self):
        return len(self.words)

    def init_state(self):
        print('creating dataset')
        for idx in tqdm(range(len(self.words))):
            # pos = self.character_pos[idx]
            word = self.words[idx]
            target_oneHot = self.target_oneHot[idx]
            encoder_input = [self.stoi['<MASK>']] * len(word)
            guess_history = [self.stoi['<SOS>']]
    
            pad_length = self.max_word_length - len(word)
            if pad_length < 0: #Word is longer than max allowed length
                pad_length = 0 #No extra pad token is required
                padded_encoder_input = encoder_input[:self.max_word_length] #word clipping is necessary
                # encoder_input_mask = torch.ones(self.max_word_length)
                encoder_input_mask = torch.zeros(self.max_word_length).bool()
            else:
                padded_encoder_input = torch.tensor(encoder_input + [self.stoi['<PAD>']] * pad_length)
                # encoder_input_mask = torch.cat((torch.ones(len(word)), torch.zeros(pad_length)), dim=0)
                encoder_input_mask = torch.cat((torch.zeros(len(word)), torch.ones(pad_length)), dim=0).bool()
    
            padded_guess_history = torch.tensor(guess_history + [self.stoi['<PAD>']] * (self.max_guesses -1))
            # guess_history_mask = torch.zeros(self.max_guesses)
            # guess_history_mask[0] = 1
            guess_history_mask = torch.ones(self.max_guesses).bool()
            guess_history_mask[0] = False

    
            self.state_list.append({'padded_encoder_input' : padded_encoder_input,
                    'encoder_input_mask' : encoder_input_mask,
                    'padded_guess_history' : padded_guess_history,
                    'guess_history_mask' : guess_history_mask,
                    'target_oneHot' : target_oneHot,
                    'gussed_characters_freq' : torch.zeros(26),
                    'word_id' : idx})
        

    def __getitem__(self, idx):
        return self.state_list[idx], torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
    
    
    
    def update_state(self, state, pred, guess_number, wordNotYetGuessed, incorrectGuesses, trials, duplicate_count): 

        ch_id = torch.argmax(pred, dim=-1)  # (batch_size,)
        ch_list = [self.itos[i.item()] for i in ch_id]

        # Vectorized updates
        trials += wordNotYetGuessed
        state['padded_guess_history'][:, guess_number] = ch_id
        state['guess_history_mask'][:, guess_number] = False

        for b, ch in enumerate(ch_list):
            if wordNotYetGuessed[b] == 1:
                # Check if this character was already guessed before
                prev_guesses = state['padded_guess_history'][b, :guess_number]
                if (prev_guesses == ch_id[b]).any():
                    duplicate_count[b] += 1
                if state['target_oneHot'][b, ch_id[b]] == 1:
                    positions = self.character_pos[state['word_id'][b]][ch]
                    for pos in positions:
                        state['padded_encoder_input'][b, pos] = ch_id[b]
                    state['target_oneHot'][b, ch_id[b]] = 0

                    if state['target_oneHot'][b].sum().item() == 0:
                        wordNotYetGuessed[b] = 0
                else:
                    incorrectGuesses[b] += 1

        return state, wordNotYetGuessed, incorrectGuesses, trials, duplicate_count


