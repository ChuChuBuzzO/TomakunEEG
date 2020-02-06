# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_rename.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(864, 586)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setEnabled(False)
        self.buttonBox.setGeometry(QtCore.QRect(490, 540, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(70, 230, 71, 111))
        font = QtGui.QFont()
        font.setFamily("MS UI Gothic")
        font.setPointSize(40)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(140, 250, 611, 31))
        self.label_2.setLineWidth(1)
        self.label_2.setObjectName("label_2")
        self.TextBrowserBefore = QtWidgets.QTextBrowser(Dialog)
        self.TextBrowserBefore.setGeometry(QtCore.QRect(20, 30, 821, 201))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TextBrowserBefore.setFont(font)
        self.TextBrowserBefore.setObjectName("TextBrowserBefore")
        self.TextBrowserAfter = QtWidgets.QTextBrowser(Dialog)
        self.TextBrowserAfter.setGeometry(QtCore.QRect(20, 330, 821, 201))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TextBrowserAfter.setFont(font)
        self.TextBrowserAfter.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByKeyboard|QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextBrowserInteraction|QtCore.Qt.TextEditable|QtCore.Qt.TextEditorInteraction|QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.TextBrowserAfter.setObjectName("TextBrowserAfter")
        self.WarningLabel = QtWidgets.QLabel(Dialog)
        self.WarningLabel.setGeometry(QtCore.QRect(160, 550, 451, 16))
        self.WarningLabel.setText("")
        self.WarningLabel.setObjectName("WarningLabel")
        self.CheckButton = QtWidgets.QPushButton(Dialog)
        self.CheckButton.setGeometry(QtCore.QRect(20, 540, 93, 28))
        self.CheckButton.setObjectName("CheckButton")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(140, 290, 391, 16))
        self.label_3.setObjectName("label_3")
        self.ConvButton = QtWidgets.QPushButton(Dialog)
        self.ConvButton.setEnabled(True)
        self.ConvButton.setGeometry(QtCore.QRect(740, 290, 93, 28))
        self.ConvButton.setObjectName("ConvButton")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "↓"))
        self.label_2.setText(_translate("Dialog", "Please rewrite the changed part. The contents of the text box above will not be reflected."))
        self.CheckButton.setText(_translate("Dialog", "Check"))
        self.label_3.setText(_translate("Dialog", "If you change it, be sure to press the Check button."))
        self.ConvButton.setText(_translate("Dialog", "Easy Paste"))
