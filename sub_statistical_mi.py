# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_statistical_mi.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(745, 974)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(390, 930, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setGeometry(QtCore.QRect(370, 840, 341, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(60, 35, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.LabelSeedCount = QtWidgets.QLabel(Dialog)
        self.LabelSeedCount.setGeometry(QtCore.QRect(230, 215, 61, 16))
        self.LabelSeedCount.setText("")
        self.LabelSeedCount.setObjectName("LabelSeedCount")
        self.PushButtonChangeChannels = QtWidgets.QPushButton(Dialog)
        self.PushButtonChangeChannels.setGeometry(QtCore.QRect(250, 595, 161, 28))
        self.PushButtonChangeChannels.setStyleSheet("background-color: lavender")
        self.PushButtonChangeChannels.setObjectName("PushButtonChangeChannels")
        self.ScrollAreaTarget = QtWidgets.QScrollArea(Dialog)
        self.ScrollAreaTarget.setGeometry(QtCore.QRect(310, 295, 381, 161))
        self.ScrollAreaTarget.setStyleSheet("")
        self.ScrollAreaTarget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.ScrollAreaTarget.setWidgetResizable(True)
        self.ScrollAreaTarget.setObjectName("ScrollAreaTarget")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 379, 159))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.ScrollAreaTarget.setWidget(self.scrollAreaWidgetContents_2)
        self.PushButtonResultPlot = QtWidgets.QPushButton(Dialog)
        self.PushButtonResultPlot.setGeometry(QtCore.QRect(90, 930, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.PushButtonResultPlot.setFont(font)
        self.PushButtonResultPlot.setStyleSheet("background-color: pink")
        self.PushButtonResultPlot.setObjectName("PushButtonResultPlot")
        self.ComboBoxFDR = QtWidgets.QComboBox(Dialog)
        self.ComboBoxFDR.setGeometry(QtCore.QRect(200, 885, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.ComboBoxFDR.setFont(font)
        self.ComboBoxFDR.setObjectName("ComboBoxFDR")
        self.ComboBoxFDR.addItem("")
        self.ComboBoxFDR.addItem("")
        self.label_22 = QtWidgets.QLabel(Dialog)
        self.label_22.setGeometry(QtCore.QRect(650, 790, 62, 15))
        self.label_22.setObjectName("label_22")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(60, 540, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.LabelProgress = QtWidgets.QLabel(Dialog)
        self.LabelProgress.setGeometry(QtCore.QRect(400, 820, 241, 20))
        self.LabelProgress.setText("")
        self.LabelProgress.setObjectName("LabelProgress")
        self.LabelTarget = QtWidgets.QLabel(Dialog)
        self.LabelTarget.setGeometry(QtCore.QRect(70, 690, 651, 20))
        self.LabelTarget.setText("")
        self.LabelTarget.setObjectName("LabelTarget")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(60, 660, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.ScrollAreaSeed = QtWidgets.QScrollArea(Dialog)
        self.ScrollAreaSeed.setGeometry(QtCore.QRect(310, 75, 381, 161))
        self.ScrollAreaSeed.setStyleSheet("")
        self.ScrollAreaSeed.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.ScrollAreaSeed.setWidgetResizable(True)
        self.ScrollAreaSeed.setObjectName("ScrollAreaSeed")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 379, 159))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.ScrollAreaSeed.setWidget(self.scrollAreaWidgetContents)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(310, 10, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: slateblue")
        self.label.setObjectName("label")
        self.label_25 = QtWidgets.QLabel(Dialog)
        self.label_25.setGeometry(QtCore.QRect(330, 545, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.LabelTargetCount = QtWidgets.QLabel(Dialog)
        self.LabelTargetCount.setGeometry(QtCore.QRect(230, 435, 61, 16))
        self.LabelTargetCount.setText("")
        self.LabelTargetCount.setObjectName("LabelTargetCount")
        self.PushButtonResult = QtWidgets.QPushButton(Dialog)
        self.PushButtonResult.setGeometry(QtCore.QRect(80, 820, 241, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.PushButtonResult.setFont(font)
        self.PushButtonResult.setStyleSheet("background-color: gold")
        self.PushButtonResult.setObjectName("PushButtonResult")
        self.DoubleSpinBoxWindowTime = QtWidgets.QDoubleSpinBox(Dialog)
        self.DoubleSpinBoxWindowTime.setGeometry(QtCore.QRect(250, 535, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.DoubleSpinBoxWindowTime.setFont(font)
        self.DoubleSpinBoxWindowTime.setObjectName("DoubleSpinBoxWindowTime")
        self.PushButtonResultSave = QtWidgets.QPushButton(Dialog)
        self.PushButtonResultSave.setGeometry(QtCore.QRect(230, 930, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.PushButtonResultSave.setFont(font)
        self.PushButtonResultSave.setStyleSheet("background-color:pink")
        self.PushButtonResultSave.setObjectName("PushButtonResultSave")
        self.ComboBoxTarget = QtWidgets.QComboBox(Dialog)
        self.ComboBoxTarget.setGeometry(QtCore.QRect(210, 250, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.ComboBoxTarget.setFont(font)
        self.ComboBoxTarget.setObjectName("ComboBoxTarget")
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setGeometry(QtCore.QRect(50, 885, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(60, 255, 271, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.SpinBoxPermTimes = QtWidgets.QSpinBox(Dialog)
        self.SpinBoxPermTimes.setGeometry(QtCore.QRect(550, 775, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.SpinBoxPermTimes.setFont(font)
        self.SpinBoxPermTimes.setObjectName("SpinBoxPermTimes")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(90, 295, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_14 = QtWidgets.QLabel(Dialog)
        self.label_14.setGeometry(QtCore.QRect(60, 910, 461, 16))
        self.label_14.setObjectName("label_14")
        self.label_21 = QtWidgets.QLabel(Dialog)
        self.label_21.setGeometry(QtCore.QRect(330, 780, 211, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(100, 75, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(60, 600, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.LabelSeed = QtWidgets.QLabel(Dialog)
        self.LabelSeed.setGeometry(QtCore.QRect(70, 630, 651, 20))
        self.LabelSeed.setText("")
        self.LabelSeed.setObjectName("LabelSeed")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(10, 730, 731, 21))
        self.label_7.setObjectName("label_7")
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(140, 470, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.DoubleSpinBoxShiftBefore = QtWidgets.QDoubleSpinBox(Dialog)
        self.DoubleSpinBoxShiftBefore.setGeometry(QtCore.QRect(370, 470, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.DoubleSpinBoxShiftBefore.setFont(font)
        self.DoubleSpinBoxShiftBefore.setObjectName("DoubleSpinBoxShiftBefore")
        self.DoubleSpinBoxShiftAfter = QtWidgets.QDoubleSpinBox(Dialog)
        self.DoubleSpinBoxShiftAfter.setGeometry(QtCore.QRect(550, 470, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.DoubleSpinBoxShiftAfter.setFont(font)
        self.DoubleSpinBoxShiftAfter.setObjectName("DoubleSpinBoxShiftAfter")
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setGeometry(QtCore.QRect(480, 480, 31, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_26 = QtWidgets.QLabel(Dialog)
        self.label_26.setGeometry(QtCore.QRect(450, 480, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(Dialog)
        self.label_27.setGeometry(QtCore.QRect(630, 480, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setGeometry(QtCore.QRect(550, 550, 41, 16))
        self.label_12.setStyleSheet("background-color:black")
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.label_15 = QtWidgets.QLabel(Dialog)
        self.label_15.setGeometry(QtCore.QRect(580, 510, 62, 15))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(Dialog)
        self.label_16.setGeometry(QtCore.QRect(350, 480, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(Dialog)
        self.label_17.setGeometry(QtCore.QRect(530, 480, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(Dialog)
        self.label_18.setGeometry(QtCore.QRect(590, 525, 16, 41))
        self.label_18.setStyleSheet("background-color: orange")
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(Dialog)
        self.label_19.setGeometry(QtCore.QRect(606, 550, 41, 16))
        self.label_19.setStyleSheet("background-color:black")
        self.label_19.setText("")
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(Dialog)
        self.label_20.setGeometry(QtCore.QRect(630, 530, 81, 16))
        self.label_20.setObjectName("label_20")
        self.label_23 = QtWidgets.QLabel(Dialog)
        self.label_23.setGeometry(QtCore.QRect(500, 530, 81, 16))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(Dialog)
        self.label_24.setGeometry(QtCore.QRect(550, 570, 62, 15))
        self.label_24.setObjectName("label_24")
        self.label_28 = QtWidgets.QLabel(Dialog)
        self.label_28.setGeometry(QtCore.QRect(570, 580, 62, 15))
        self.label_28.setObjectName("label_28")
        self.label_29 = QtWidgets.QLabel(Dialog)
        self.label_29.setGeometry(QtCore.QRect(640, 590, 62, 15))
        self.label_29.setObjectName("label_29")
        self.label_30 = QtWidgets.QLabel(Dialog)
        self.label_30.setGeometry(QtCore.QRect(620, 600, 62, 15))
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(Dialog)
        self.label_31.setGeometry(QtCore.QRect(620, 570, 91, 16))
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(Dialog)
        self.label_32.setGeometry(QtCore.QRect(30, 210, 271, 16))
        self.label_32.setObjectName("label_32")
        self.label_33 = QtWidgets.QLabel(Dialog)
        self.label_33.setGeometry(QtCore.QRect(30, 430, 271, 16))
        self.label_33.setObjectName("label_33")
        self.label_34 = QtWidgets.QLabel(Dialog)
        self.label_34.setGeometry(QtCore.QRect(340, 755, 551, 16))
        self.label_34.setObjectName("label_34")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", "●Seed raw： this one"))
        self.PushButtonChangeChannels.setText(_translate("Dialog", "ChangeChannels"))
        self.PushButtonResultPlot.setText(_translate("Dialog", "PlotResult"))
        self.ComboBoxFDR.setItemText(0, _translate("Dialog", "Yes"))
        self.ComboBoxFDR.setItemText(1, _translate("Dialog", "No"))
        self.label_22.setText(_translate("Dialog", "times"))
        self.label_6.setText(_translate("Dialog", "●Window time："))
        self.label_9.setText(_translate("Dialog", "●Target Channels："))
        self.label.setText(_translate("Dialog", "MI_Calculation"))
        self.label_25.setText(_translate("Dialog", "s"))
        self.PushButtonResult.setText(_translate("Dialog", "Calculation"))
        self.PushButtonResultSave.setText(_translate("Dialog", "SaveResult"))
        self.label_13.setText(_translate("Dialog", "FDR Correction："))
        self.label_3.setText(_translate("Dialog", "●Target raw："))
        self.label_5.setText(_translate("Dialog", "： Target event name："))
        self.label_14.setText(_translate("Dialog", "If channels are increased, FDR correction will be applied accordingly."))
        self.label_21.setText(_translate("Dialog", "●Permutation Times："))
        self.label_4.setText(_translate("Dialog", "： Seed event name："))
        self.label_8.setText(_translate("Dialog", "●Seed Channels："))
        self.label_7.setText(_translate("Dialog", "The Seed raw event is fixed, and the area for the window time from it is shifted "))
        self.label_10.setText(_translate("Dialog", "： shift length"))
        self.label_11.setText(_translate("Dialog", "～"))
        self.label_26.setText(_translate("Dialog", "s"))
        self.label_27.setText(_translate("Dialog", "s"))
        self.label_15.setText(_translate("Dialog", "event"))
        self.label_16.setText(_translate("Dialog", "-"))
        self.label_17.setText(_translate("Dialog", "＋"))
        self.label_20.setText(_translate("Dialog", "shift length"))
        self.label_23.setText(_translate("Dialog", "shift length"))
        self.label_24.setText(_translate("Dialog", "-----"))
        self.label_28.setText(_translate("Dialog", "-----"))
        self.label_29.setText(_translate("Dialog", "-----"))
        self.label_30.setText(_translate("Dialog", "-----"))
        self.label_31.setText(_translate("Dialog", "Window time"))
        self.label_32.setText(_translate("Dialog", "If multiple events exist, use only first one"))
        self.label_33.setText(_translate("Dialog", "If multiple events exist, use only first one"))
        self.label_34.setText(_translate("Dialog", "and tested with the area moved around the Target raw event."))
