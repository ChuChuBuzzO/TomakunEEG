# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_seed_target.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(837, 889)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(320, 860, 201, 32))
        self.buttonBox.setStyleSheet("background-color: lightblue")
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(160, 10, 62, 15))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(600, 0, 62, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.PushButtonSelectAllSeed = QtWidgets.QPushButton(Dialog)
        self.PushButtonSelectAllSeed.setGeometry(QtCore.QRect(20, 860, 93, 28))
        self.PushButtonSelectAllSeed.setObjectName("PushButtonSelectAllSeed")
        self.PushButtonClearSeed = QtWidgets.QPushButton(Dialog)
        self.PushButtonClearSeed.setGeometry(QtCore.QRect(120, 860, 93, 28))
        self.PushButtonClearSeed.setObjectName("PushButtonClearSeed")
        self.PushButtonClearTarget = QtWidgets.QPushButton(Dialog)
        self.PushButtonClearTarget.setGeometry(QtCore.QRect(720, 860, 93, 28))
        self.PushButtonClearTarget.setObjectName("PushButtonClearTarget")
        self.PushButtonSelectAllTarget = QtWidgets.QPushButton(Dialog)
        self.PushButtonSelectAllTarget.setGeometry(QtCore.QRect(620, 860, 93, 28))
        self.PushButtonSelectAllTarget.setObjectName("PushButtonSelectAllTarget")
        self.gridLayoutWidget = QtWidgets.QWidget(Dialog)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 30, 381, 831))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayoutSeed = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayoutSeed.setContentsMargins(0, 0, 0, 0)
        self.gridLayoutSeed.setObjectName("gridLayoutSeed")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(Dialog)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(440, 30, 381, 831))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayoutTarget = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayoutTarget.setContentsMargins(0, 0, 0, 0)
        self.gridLayoutTarget.setObjectName("gridLayoutTarget")
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(400, 40, 20, 811))
        self.line.setStyleSheet("")
        self.line.setLineWidth(3)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Seed"))
        self.label_2.setText(_translate("Dialog", "Target"))
        self.PushButtonSelectAllSeed.setText(_translate("Dialog", "Select All"))
        self.PushButtonClearSeed.setText(_translate("Dialog", "Clear"))
        self.PushButtonClearTarget.setText(_translate("Dialog", "Clear"))
        self.PushButtonSelectAllTarget.setText(_translate("Dialog", "Select All"))
