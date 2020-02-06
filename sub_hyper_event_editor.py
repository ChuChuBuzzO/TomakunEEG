# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_hyper_event_editor.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1223, 929)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(860, 890, 341, 32))
        self.buttonBox.setStyleSheet("")
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.PushButtonADDEvt = QtWidgets.QPushButton(Dialog)
        self.PushButtonADDEvt.setGeometry(QtCore.QRect(720, 890, 93, 28))
        self.PushButtonADDEvt.setStyleSheet("background-color:pink")
        self.PushButtonADDEvt.setObjectName("PushButtonADDEvt")
        self.PushButtonDeleteEvt = QtWidgets.QPushButton(Dialog)
        self.PushButtonDeleteEvt.setGeometry(QtCore.QRect(830, 890, 93, 28))
        self.PushButtonDeleteEvt.setStyleSheet("background-color: lavender")
        self.PushButtonDeleteEvt.setObjectName("PushButtonDeleteEvt")
        self.TextEdit = QtWidgets.QTextEdit(Dialog)
        self.TextEdit.setGeometry(QtCore.QRect(220, 890, 481, 31))
        font = QtGui.QFont()
        font.setItalic(True)
        self.TextEdit.setFont(font)
        self.TextEdit.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.TextEdit.setStyleSheet("background-color: linen")
        self.TextEdit.setObjectName("TextEdit")
        self.PushButtonUpdateState = QtWidgets.QPushButton(Dialog)
        self.PushButtonUpdateState.setGeometry(QtCore.QRect(30, 890, 93, 28))
        self.PushButtonUpdateState.setStyleSheet("background-color:lightgreen")
        self.PushButtonUpdateState.setObjectName("PushButtonUpdateState")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(420, 20, 391, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(20, 50, 1181, 821))
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayoutLeft = QtWidgets.QVBoxLayout()
        self.verticalLayoutLeft.setObjectName("verticalLayoutLeft")
        self.horizontalLayout_2.addLayout(self.verticalLayoutLeft)
        self.verticalLayoutMiddle = QtWidgets.QVBoxLayout()
        self.verticalLayoutMiddle.setObjectName("verticalLayoutMiddle")
        self.horizontalLayout_2.addLayout(self.verticalLayoutMiddle)
        self.verticalLayoutRight = QtWidgets.QVBoxLayout()
        self.verticalLayoutRight.setObjectName("verticalLayoutRight")
        self.horizontalLayout_2.addLayout(self.verticalLayoutRight)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.PushButtonADDEvt.setText(_translate("Dialog", "ADD_Event"))
        self.PushButtonDeleteEvt.setText(_translate("Dialog", "Delete_Event"))
        self.PushButtonUpdateState.setText(_translate("Dialog", "UpdateState"))
        self.label.setText(_translate("Dialog", "*Comments / onset_time(s) / duraion(s) の順"))
