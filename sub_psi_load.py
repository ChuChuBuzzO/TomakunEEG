# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_psi_load.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(752, 214)
        self.PushButtonContinuousMode = QtWidgets.QPushButton(Dialog)
        self.PushButtonContinuousMode.setGeometry(QtCore.QRect(250, 110, 251, 91))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.PushButtonContinuousMode.setFont(font)
        self.PushButtonContinuousMode.setStyleSheet("background-color: greenyellow")
        self.PushButtonContinuousMode.setObjectName("PushButtonContinuousMode")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(140, 60, 461, 41))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(180, 30, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.PushButtonLoadFile = QtWidgets.QPushButton(Dialog)
        self.PushButtonLoadFile.setGeometry(QtCore.QRect(430, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.PushButtonLoadFile.setFont(font)
        self.PushButtonLoadFile.setStyleSheet("background-color: pink")
        self.PushButtonLoadFile.setObjectName("PushButtonLoadFile")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.PushButtonContinuousMode.setText(_translate("Dialog", "ContinuousPlot"))
        self.label.setText(_translate("Dialog", "Load Continuous Directory"))
        self.PushButtonLoadFile.setText(_translate("Dialog", "Load"))
