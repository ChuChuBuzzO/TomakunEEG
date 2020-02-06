% report.txtを処理した落とすべきコンポーネントの配列
try
    rep = csvread("rep_a.csv");

    % これらをeegからremoveする。    
    EEG = pop_subcomp(EEG, rep);

    EEG = pop_saveset(EEG, 'test_after.set', './');

catch
    EEG = pop_saveset(EEG, 'test_after.set', './');

end
% EEG = pop_subcomp(EEG, [1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25])
% 空配列がrepに渡ってしまうとエラーになるのでtryで回避しとく
% EEG = pop_saveset(EEG, 'test_after.set', './');

% save file. test_AAという名前で保存
%EEG = pop_saveset(EEG, 'test_AA.set', './');