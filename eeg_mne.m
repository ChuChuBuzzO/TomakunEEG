% [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
 
% ch数, ced_pathもtextファイルとして受け渡しすることにした(2020/1/9)
[ch_num, ced_path] = textread("pass.txt", '%u %s');

% 今回26chのlocationは、animallab.cedとして保存しておいた。

% EEGデータの読み込み
EEG = pop_loadset('pass.set', './');

% 必要なチャネルのselect。
EEG = pop_select(EEG, 'channel', 1:ch_num);

% channel locationを読み込む
EEG.chanlocs = pop_chanedit(EEG.chanlocs, 'load', {char(ced_path), 'filetype', 'autodetect'});

% run_ica
EEG = pop_runica(EEG, 'runica');

% これないとなぜか下でクリックできなくなる。
eeglab redraw;

% Adjust algorithm。結果まで出すとこの処理終わっちゃうから、すぐ次に行っておかしくなる。
% このreportファイルを読み取って、component落とすのが妥当。
try
    EEG = interface_ADJ(EEG, "report.txt");
catch exception
    EEG = pop_saveset(EEG, 'test_after.set', './');
    throw(exception);
end