# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sub_eegview_pyqtgraph.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1730, 917)
        Dialog.setStyleSheet("background-color: white")
        self.graphicsView = GraphicsLayoutWidget(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 1711, 851))
        self.graphicsView.setObjectName("graphicsView")
        self.PushButtonEvents = QtWidgets.QPushButton(Dialog)
        self.PushButtonEvents.setGeometry(QtCore.QRect(10, 880, 111, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PushButtonEvents.setFont(font)
        self.PushButtonEvents.setStyleSheet("background-color: rgb(229, 204, 255)")
        self.PushButtonEvents.setObjectName("PushButtonEvents")
        self.PushButtonOption = QtWidgets.QPushButton(Dialog)
        self.PushButtonOption.setGeometry(QtCore.QRect(10, 840, 71, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PushButtonOption.setFont(font)
        self.PushButtonOption.setStyleSheet("background-color: rgb(229, 204, 255)")
        self.PushButtonOption.setObjectName("PushButtonOption")
        self.horizontalScrollBar = QtWidgets.QScrollBar(Dialog)
        self.horizontalScrollBar.setGeometry(QtCore.QRect(140, 840, 1551, 31))
        self.horizontalScrollBar.setStyleSheet("background-color: gray")
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.verticalScrollBar = QtWidgets.QScrollBar(Dialog)
        self.verticalScrollBar.setGeometry(QtCore.QRect(1710, 10, 16, 781))
        self.verticalScrollBar.setStyleSheet("background-color: gray")
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.PushButtonEventsEdit = QtWidgets.QPushButton(Dialog)
        self.PushButtonEventsEdit.setGeometry(QtCore.QRect(140, 880, 111, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.PushButtonEventsEdit.setFont(font)
        self.PushButtonEventsEdit.setStyleSheet("background-color: rgb(229, 204, 255)")
        self.PushButtonEventsEdit.setObjectName("PushButtonEventsEdit")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.PushButtonEvents.setText(_translate("Dialog", "Events_check"))
        self.PushButtonOption.setText(_translate("Dialog", "Option"))
        self.PushButtonEventsEdit.setText(_translate("Dialog", "Events_edit"))
from pyqtgraph import GraphicsLayoutWidget
