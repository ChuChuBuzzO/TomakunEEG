% report.txt�������������Ƃ��ׂ��R���|�[�l���g�̔z��
try
    rep = csvread("rep_a.csv");

    % ������eeg����remove����B    
    EEG = pop_subcomp(EEG, rep);

    EEG = pop_saveset(EEG, 'test_after.set', './');

catch
    EEG = pop_saveset(EEG, 'test_after.set', './');

end
% EEG = pop_subcomp(EEG, [1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25])
% ��z��rep�ɓn���Ă��܂��ƃG���[�ɂȂ�̂�try�ŉ�����Ƃ�
% EEG = pop_saveset(EEG, 'test_after.set', './');

% save file. test_AA�Ƃ������O�ŕۑ�
%EEG = pop_saveset(EEG, 'test_AA.set', './');