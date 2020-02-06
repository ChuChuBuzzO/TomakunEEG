# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_sub_option.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(40, 260, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 70, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(30, 160, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.spinBox_c = QtWidgets.QSpinBox(Dialog)
        self.spinBox_c.setGeometry(QtCore.QRect(270, 70, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.spinBox_c.setFont(font)
        self.spinBox_c.setObjectName("spinBox_c")
        self.spinBox_s = QtWidgets.QSpinBox(Dialog)
        self.spinBox_s.setGeometry(QtCore.QRect(270, 160, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.spinBox_s.setFont(font)
        self.spinBox_s.setObjectName("spinBox_s")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Viewing Channels num"))
        self.label_2.setText(_translate("Dialog", "Viewing time length (s)"))
