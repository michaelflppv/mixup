/***************************************************************************
*                                                                          *
*   Copyright (C) 2018 by David B. Blumenthal                              *
*                                                                          *
*   This file is part of GEDLIB.                                           *
*                                                                          *
*   GEDLIB is free software: you can redistribute it and/or modify it      *
*   under the terms of the GNU Lesser General Public License as published  *
*   by the Free Software Foundation, either version 3 of the License, or   *
*   (at your option) any later version.                                    *
*                                                                          *
*   GEDLIB is distributed in the hope that it will be useful,              *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of         *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the           *
*   GNU Lesser General Public License for more details.                    *
*                                                                          *
*   You should have received a copy of the GNU Lesser General Public       *
*   License along with GEDLIB. If not, see <http://www.gnu.org/licenses/>. *
*                                                                          *
***************************************************************************/

/*!
 * @file chem_2.hpp
 * @brief ged::CHEM2 class declaration.
 */

#ifndef SRC_EDIT_COSTS_CHEM_2_HPP_
#define SRC_EDIT_COSTS_CHEM_2_HPP_

#include "edit_costs.hpp"

namespace ged {

/*!
 * @brief Edit cost functions for chemical graphs such as the ones contained in the datasets Mutagenicity, AIDS, PAH, MAO, Alkane, and Acyclic.
 * @details Cost function for graphs that represent chemical compounds.
 * Nodes are expected to be attributed with a chemical symbol (named "chem").
 * Edges are expected to be attributed with their valence (named "valence").
 * The datasets pah, mao, alkane, and acyclic are part of GREYC's chemistry dataset which can be downloaded from https://brunl01.users.greyc.fr/CHEMISTRY/.
 * The datasets Mutagenicity and AIDS are contained in the IAM graph database repository which can be downloaded from http://www.fki.inf.unibe.ch/databases/iam-graph-database:
 * - K. Riesen, H. Bunke:
 *   &ldquo;IAM graph database repository for graph based pattern recognition and machine learning&rdquo;,
 *   https://doi.org/10.1007/978-3-540-89689-0_33
 *
 * Implements the edit costs suggested in:
 * - Z. Abu-Aisheh, R. Raveaux, J.-Y. Ramel:
 *   &ldquo;A graph database repository and performance evaluation metrics for graph edit distance&rdquo;,
 *   https://doi.org/10.1007/978-3-319-18224-7_14
 * - K. Riesen, H. Bunke:
 *   &ldquo;Graph data&rdquo;, *Graph Classification and Clustering Based on Vector Space Embedding*,
 *   https://doi.org/10.1142/9789814304726_0004
 */
template<class UserNodeLabel, class UserEdgeLabel>
class CHEM2 : public EditCosts<UserNodeLabel, UserEdgeLabel> {
public:

	virtual ~CHEM2();

	/*!
	 * @brief Constructor.
	 * @param[in] node_ins_del_cost Cost for deleting or inserting nodes.
	 * @param[in] edge_ins_del_cost Cost for deleting or inserting edges.
	 * @param[in] alpha Importance of node edit operations vs. importance of edge edit operations.
	 * @note Calling the constructor with the default arguments constructs the edit costs suggested in https://doi.org/10.1007/978-3-319-18224-7_14.
	 */
	CHEM2(double node_ins_del_cost = 11, double edge_ins_del_cost = 1.1, double alpha = 0.25);

	virtual double node_ins_cost_fun(const UserNodeLabel & node_label) const final;

	virtual double node_del_cost_fun(const UserNodeLabel & node_label) const final;

	virtual double node_rel_cost_fun(const UserNodeLabel & node_label_1, const UserNodeLabel & node_label_2) const final;

	virtual UserNodeLabel median_node_label(const std::vector<UserNodeLabel> & node_labels) const final;

	virtual double edge_ins_cost_fun(const UserEdgeLabel & edge_label) const final;

	virtual double edge_del_cost_fun(const UserEdgeLabel & edge_label) const final;

	virtual double edge_rel_cost_fun(const UserEdgeLabel & edge_label_1, const UserEdgeLabel & edge_label_2) const final;

	virtual UserEdgeLabel median_edge_label(const std::vector<UserEdgeLabel> & edge_labels) const final;

private:

	double node_ins_del_cost_;

	double edge_ins_del_cost_;

	double alpha_;
};

}

#include "chem_2.ipp"

#endif /* SRC_EDIT_COSTS_CHEM_2_HPP_ */
