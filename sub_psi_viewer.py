# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_psi_viewer.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(695, 651)
        Dialog.setStyleSheet("background-color: white")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(470, 610, 211, 32))
        self.buttonBox.setStyleSheet("background-color: lightblue")
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.PushButtonPrev_2 = QtWidgets.QPushButton(Dialog)
        self.PushButtonPrev_2.setGeometry(QtCore.QRect(290, 610, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PushButtonPrev_2.setFont(font)
        self.PushButtonPrev_2.setStyleSheet("background-color: lightgreen")
        self.PushButtonPrev_2.setObjectName("PushButtonPrev_2")
        self.horizontalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 671, 591))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.PushButtonPrev_2.setText(_translate("Dialog", "detail"))
