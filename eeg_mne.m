% [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
 
% ch��, ced_path��text�t�@�C���Ƃ��Ď󂯓n�����邱�Ƃɂ���(2020/1/9)
[ch_num, ced_path] = textread("pass.txt", '%u %s');

% ����26ch��location�́Aanimallab.ced�Ƃ��ĕۑ����Ă������B

% EEG�f�[�^�̓ǂݍ���
EEG = pop_loadset('pass.set', './');

% �K�v�ȃ`���l����select�B
EEG = pop_select(EEG, 'channel', 1:ch_num);

% channel location��ǂݍ���
EEG.chanlocs = pop_chanedit(EEG.chanlocs, 'load', {char(ced_path), 'filetype', 'autodetect'});

% run_ica
EEG = pop_runica(EEG, 'runica');

% ����Ȃ��ƂȂ������ŃN���b�N�ł��Ȃ��Ȃ�B
eeglab redraw;

% Adjust algorithm�B���ʂ܂ŏo���Ƃ��̏����I������Ⴄ����A�������ɍs���Ă��������Ȃ�B
% ����report�t�@�C����ǂݎ���āAcomponent���Ƃ��̂��Ó��B
try
    EEG = interface_ADJ(EEG, "report.txt");
catch exception
    EEG = pop_saveset(EEG, 'test_after.set', './');
    throw(exception);
end