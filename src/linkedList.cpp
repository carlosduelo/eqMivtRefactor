/*
Author: Carlos Duelo Serrano 
Company: Cesvima

Notes:

 */

#include <linkedList.h>

#include <iostream>

namespace eqMivt
{
LinkedList::LinkedList()
{
	list = 0;
	last = 0;
	memoryList = 0;
	freePositions = 0;
}

void LinkedList::reSize(int size)
{
	if (memoryList != 0)
		delete[] memoryList;

	freePositions 	= size;
	memoryList 	= new NodeLinkedList[size];
	list 		= memoryList;
	last 		= &memoryList[size-1];

	for(int i=0; i<size; i++)
	{
		if (i==0)
		{
			memoryList[i].after 	= &memoryList[i+1];
			memoryList[i].before 	= 0;
			memoryList[i].element 	= i;
			memoryList[i].id		= 0;
			memoryList[i].refs		= 0;
		}
		else if (i==size-1)
		{
			memoryList[i].after 	= 0;
			memoryList[i].before 	= &memoryList[i-1];
			memoryList[i].element 	= i;
			memoryList[i].id		= 0;
			memoryList[i].refs		= 0;
		}
		else
		{
			memoryList[i].after 	= &memoryList[i+1];
			memoryList[i].before 	= &memoryList[i-1];
			memoryList[i].element 	= i;
			memoryList[i].id		= 0;
			memoryList[i].refs		= 0;
		}
	}
}

LinkedList::~LinkedList()
{
	if (memoryList != 0)
		delete[] memoryList;
}


NodeLinkedList * LinkedList::getFirstFreePosition()
{
	NodeLinkedList * first = list;

	if (freePositions == 0)
	{
		// Search first free position
		while(list->refs != 0)
		{
			moveToLastPosition(list);
			if (first == list)
			{
				return 0;
			}
		}
	}
	else
	{
		// Search first free position
		while(list->refs != 0 && list->id > 0)
		{
			moveToLastPosition(list);
			if (first == list)
			{
				std::cerr<<"Error cache is unistable"<<std::endl;
				throw;
			}
		}

		freePositions--;
	}
	return list;
}

NodeLinkedList * LinkedList::moveToLastPosition(NodeLinkedList * node)
{
	if (node == list)
	{
		NodeLinkedList * first = list;

		list = first->after;
		list->before = 0;
		
		first->after  = 0;
		first->before = last;
		
		last->after = first;
		
		last = first;

		return first;
	}
	else if (node == last)
	{
		return node;
	}
	else
	{
		node->before->after = node->after;
		node->after->before = node->before;
		
		last->after = node;
		
		node->before = last;
		node->after  = 0;
		last = node;
		
		return node;
	}
}
}
