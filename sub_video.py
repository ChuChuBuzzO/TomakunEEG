# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_video.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(993, 566)
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 20, 841, 481))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.SliderVideo = QtWidgets.QSlider(Dialog)
        self.SliderVideo.setEnabled(True)
        self.SliderVideo.setGeometry(QtCore.QRect(100, 521, 771, 31))
        self.SliderVideo.setOrientation(QtCore.Qt.Horizontal)
        self.SliderVideo.setObjectName("SliderVideo")
        self.PushButtonPlay = QtWidgets.QPushButton(Dialog)
        self.PushButtonPlay.setGeometry(QtCore.QRect(30, 510, 61, 51))
        self.PushButtonPlay.setText("")
        self.PushButtonPlay.setObjectName("PushButtonPlay")
        self.SliderVolume = QtWidgets.QSlider(Dialog)
        self.SliderVolume.setGeometry(QtCore.QRect(890, 340, 22, 160))
        self.SliderVolume.setMaximum(100)
        self.SliderVolume.setOrientation(QtCore.Qt.Vertical)
        self.SliderVolume.setObjectName("SliderVolume")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(890, 300, 41, 31))
        self.label.setText("")
        self.label.setObjectName("label")
        self.TimeDisp = QtWidgets.QLabel(Dialog)
        self.TimeDisp.setGeometry(QtCore.QRect(880, 530, 101, 16))
        self.TimeDisp.setText("")
        self.TimeDisp.setObjectName("TimeDisp")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
