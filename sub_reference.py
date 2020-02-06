# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_reference.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(100, 50, 231, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(70, 90, 261, 161))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.PushButtonNoRef = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.PushButtonNoRef.setObjectName("PushButtonNoRef")
        self.verticalLayout.addWidget(self.PushButtonNoRef)
        self.PushButtonAverage = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.PushButtonAverage.setObjectName("PushButtonAverage")
        self.verticalLayout.addWidget(self.PushButtonAverage)
        self.PushButtonSElectrode = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.PushButtonSElectrode.setObjectName("PushButtonSElectrode")
        self.verticalLayout.addWidget(self.PushButtonSElectrode)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Choose the Reference"))
        self.PushButtonNoRef.setText(_translate("Dialog", "No Reference (+N)"))
        self.PushButtonAverage.setText(_translate("Dialog", "Average (+A)"))
        self.PushButtonSElectrode.setText(_translate("Dialog", "Single or multiple electrodes (+M)"))

